from __future__ import print_function

import os
import time
import argparse
import logging
import hashlib
import copy

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn

import sparselearning
from sparselearning.core import Masking, CosineDecay, LinearDecay, normalize_weight, reparameterization_update, post_update
from sparselearning.models import AlexNet, VGG16, LeNet_300_100, LeNet_5_Caffe, WideResNet, MLP_CIFAR10, ResNet34, ResNet18
from sparselearning.utils import get_mnist_dataloaders, get_cifar10_dataloaders, get_cifar100_dataloaders
import torchvision
import torchvision.transforms as transforms
import warnings
import wandb
from npb import *
warnings.filterwarnings("ignore", category=UserWarning)
cudnn.benchmark = True
cudnn.deterministic = True
# os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,4,5,6,7"
if not os.path.exists('./models'): os.mkdir('./models')
if not os.path.exists('./logs'): os.mkdir('./logs')
logger = None

models = {}
models['MLPCIFAR10'] = (MLP_CIFAR10,[])
models['lenet5'] = (LeNet_5_Caffe,[])
models['lenet300-100'] = (LeNet_300_100,[])
models['ResNet34'] = ()
models['ResNet18'] = ()
models['alexnet-s'] = (AlexNet, ['s', 10])
models['alexnet-b'] = (AlexNet, ['b', 10])
models['vgg-c'] = (VGG16, ['C', 10])
models['vgg-d'] = (VGG16, ['D', 10])
models['vgg-like'] = (VGG16, ['like', 10])
models['wrn-28-2'] = (WideResNet, [28, 2, 10, 0.3])
models['wrn-22-8'] = (WideResNet, [22, 8, 10, 0.3])
models['wrn-16-8'] = (WideResNet, [16, 8, 10, 0.3])
models['wrn-16-10'] = (WideResNet, [16, 10, 10, 0.3])

def setup_logger(args):
    global logger
    if logger == None:
        logger = logging.getLogger()
    else:  # wish there was a logger.close()
        for handler in logger.handlers[:]:  # make a copy of the list
            logger.removeHandler(handler)

    args_copy = copy.deepcopy(args)
    # copy to get a clean hash
    # use the same log file hash if iterations or verbose are different
    # these flags do not change the results
    args_copy.iters = 1
    args_copy.verbose = False
    args_copy.log_interval = 1
    args_copy.seed = 0

    log_path = './logs/{0}_{1}_{2}.log'.format(args.model, args.density, hashlib.md5(str(args_copy).encode('utf-8')).hexdigest()[:8])

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt='%(asctime)s: %(message)s', datefmt='%H:%M:%S')

    fh = logging.FileHandler(log_path)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

def print_and_log(msg):
    global logger
    print(msg)
    logger.info(msg)

def train(args, model, device, train_loader, optimizer, epoch, mask=None):
    model.train()
    train_loss = 0
    correct = 0
    n = 0
    enabled = False
    if args.amp:
        enabled = True
        torch.backends.cudnn.benchmark = True
        scaler = torch.cuda.amp.GradScaler(enabled=True)

    for batch_idx, (data, target) in enumerate(train_loader):

        data, target = data.to(device), target.to(device)
        if args.fp16: data = data.half()
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=enabled):
            output = model(data)
            loss = F.nll_loss(output, target)

        train_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        n += target.shape[0]

        # if args.fp16:
        #     optimizer.backward(loss)
        # else:
        #     loss.backward()
        if args.amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()


    # training summary
    train_acc = 100. * correct / float(n)
    train_loss = train_loss/batch_idx

    print_and_log('\n Training summary: Epoch: {}, Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%) \n'.format(
        epoch , train_loss, correct, n, train_acc))
    if args.wandb:
        wandb.log({'train acc': train_acc, 'train loss': train_loss, 'epoch':epoch})

def evaluate(args, model, device, test_loader, is_test_set=False):
    model.eval()
    test_loss = 0
    correct = 0
    n = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            if args.fp16: data = data.half()
            model.t = target
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            n += target.shape[0]

    test_loss /= float(n)

    print_and_log('\n{}: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
        'Test evaluation' if is_test_set else 'Evaluation',
        test_loss, correct, n, 100. * correct / float(n)))
    return correct / float(n)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')

    parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                        help='input batch size for training (default: 100)')
    parser.add_argument('--test-batch-size', type=int, default=100, metavar='N',
                        help='input batch size for testing (default: 100)')
    parser.add_argument('--multiplier', type=int, default=1, metavar='N',
                        help='extend training time by multiplier times')
    parser.add_argument('--epochs', type=int, default=250, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--lr_score', type=float, default=0.1, metavar='LR',
                        help='learning rate score (default: 0.1)', required=False)
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=17, metavar='S', help='random seed (default: 17)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--optimizer', type=str, default='sgd', help='The optimizer to use. Default: sgd. Options: sgd, adam.')
    randomhash = ''.join(str(time.time()).split('.'))
    parser.add_argument('--save', type=str, default=f'cm/archive/quannt40/dst/checkpoints/{randomhash}.pt',
                        help='path to save the final model')
    parser.add_argument('--data', type=str, default='mnist')
    parser.add_argument('--decay_frequency', type=int, default=25000)
    parser.add_argument('--l1', type=float, default=0.0)
    parser.add_argument('--lr_drop_rate', type=float, default=0.1)
    parser.add_argument('--fp16', action='store_true', help='Run in fp16 mode.')
    parser.add_argument('--amp', action='store_true', help='torch mix precision.')
    parser.add_argument('--valid_split', type=float, default=0.1)
    parser.add_argument('--resume', type=str)
    parser.add_argument('--start-epoch', type=int, default=1)
    parser.add_argument('--model', type=str, default='')
    parser.add_argument('--l2', type=float, default=5.0e-4)
    parser.add_argument('--iters', type=int, default=1, help='How many times the model should be run after each other. Default=1')
    parser.add_argument('--save-features', action='store_true', help='Resumes a saved model and saves its feature data to disk for plotting.')
    parser.add_argument('--bench', action='store_true', help='Enables the benchmarking of layers and estimates sparse speedups')
    parser.add_argument('--max-threads', type=int, default=10, help='How many threads to use for data loading.')
    parser.add_argument('--wandb', action='store_true', help='logging results with wandb')
    parser.add_argument('--device', type=str, default='cuda:0')
    # ITOP settings
    sparselearning.core.add_sparse_args(parser)

    args = parser.parse_args()
    setup_logger(args)
    print_and_log(args)

    if args.fp16:
        try:
            from apex.fp16_utils import FP16_Optimizer
        except:
            print('WARNING: apex not installed, ignoring --fp16 option')
            args.fp16 = False

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    # device = torch.device("cuda" if use_cuda else "cpu")
    device = args.device

    print_and_log('\n\n')
    print_and_log('='*80)
    torch.manual_seed(args.seed)

    if args.wandb:
        wandb.login(key='74ac7eba00fea7e805a70861a86c7767406946c9')
        name = f'{args.model}_{args.data}_{args.method}_alpha_{args.alpha}_beta_{args.beta}'
        run = wandb.init(
            # Set the project where this run will be logged
            project='DST',
            name=name,
            # Track hyperparameters and run metadata
            config=args
        )
    args.save_path = 'cm/archive/quannt40/dst'
    for i in range(args.iters):
        print_and_log("\nIteration start: {0}/{1}\n".format(i+1, args.iters))

        if args.data == 'mnist':
            train_loader, valid_loader, test_loader = get_mnist_dataloaders(args, validation_split=args.valid_split)
        elif args.data == 'cifar10':
            train_loader, valid_loader, test_loader = get_cifar10_dataloaders(args, args.valid_split, max_threads=args.max_threads)
            c = 10
        elif args.data == 'cifar100':
            train_loader, valid_loader, test_loader = get_cifar100_dataloaders(args, args.valid_split, max_threads=args.max_threads)
            c = 100
        if args.model not in models:
            print('You need to select an existing model via the --model argument. Available models include: ')
            for key in models:
                print('\t{0}'.format(key))
            raise Exception('You need to select a model')
        elif args.model == 'ResNet18':
            model = ResNet18(c=c).to(device)
        elif args.model == 'ResNet34':
            model = ResNet34(c=c).to(device)
        else:
            cls, cls_args = models[args.model]
            model = cls(*(cls_args + [args.save_features, args.bench])).to(device)
        print_and_log(model)
        print_and_log('=' * 60)
        print_and_log(args.model)
        print_and_log('=' * 60)

        print_and_log('=' * 60)
        print_and_log('Prune mode: {0}'.format(args.death))
        print_and_log('Growth mode: {0}'.format(args.growth))
        print_and_log('Redistribution mode: {0}'.format(args.redistribution))
        print_and_log('=' * 60)


        optimizer = None
        if args.optimizer == 'sgd':
            optimizer = optim.SGD(model.parameters(),lr=args.lr,momentum=args.momentum,weight_decay=args.l2, nesterov=True)
        elif args.optimizer == 'adam':
            optimizer = optim.Adam(model.parameters(),lr=args.lr,weight_decay=args.l2)
        else:
            print('Unknown optimizer: {0}'.format(args.optimizer))
            raise Exception('Unknown optimizer.')

        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(args.epochs / 2) * args.multiplier, int(args.epochs * 3 / 4) * args.multiplier], gamma=args.lr_drop_rate, last_epoch=-1)


        if args.resume:
            if os.path.isfile(args.resume):
                print_and_log("=> loading checkpoint '{}'".format(args.resume))
                checkpoint = torch.load(args.resume)
                args.start_epoch = checkpoint['epoch']
                model.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                print_and_log("=> loaded checkpoint '{}' (epoch {})"
                      .format(args.resume, checkpoint['epoch']))
                print_and_log('Testing...')
                evaluate(args, model, device, test_loader)
                model.feats = []
                model.densities = []
                plot_class_feature_histograms(args, model, device, train_loader, optimizer)
            else:
                print_and_log("=> no checkpoint found at '{}'".format(args.resume))


        if args.fp16:
            print('FP16')
            optimizer = FP16_Optimizer(optimizer,
                                       static_loss_scale = None,
                                       dynamic_loss_scale = True,
                                       dynamic_loss_args = {'init_scale': 2 ** 16})
            model = model.half()

        mask = None
        # if args.sparse:
        #     decay = CosineDecay(args.death_rate, len(train_loader)*(args.epochs*args.multiplier))
        #     mask = Masking(optimizer, death_rate=args.death_rate, death_mode=args.death, death_rate_decay=decay, growth_mode=args.growth,
        #                    redistribution_mode=args.redistribution, args=args)
        #     mask.add_module(model, sparse_init=args.sparse_init, density=args.density)

        print('Pruning at Initialization')
        NPB_register(model, args)
        model.apply(lambda m: setattr(m, "npb", True))
        for iter in range(100):
            eff_paths, eff_nodes = NPB_objective(model, args)
            print(f'eff nodes: {eff_nodes}, eff paths: {eff_paths}')
            if args.wandb:
                wandb.log({'eff nodes': eff_nodes, 'eff paths': eff_paths})
        model.apply(lambda m: setattr(m, "npb", False))

        best_acc = 0.0


        for epoch in range(1, args.epochs*args.multiplier + 1):
            t0 = time.time()
            train(args, model, device, train_loader, optimizer, epoch, mask)
            lr_scheduler.step()
            if args.valid_split > 0.0:
                val_acc = evaluate(args, model, device, valid_loader)

            if val_acc > best_acc:
                print('Saving model')
                best_acc = val_acc
                torch.save(model.state_dict(), args.save)

            print_and_log('Current learning rate: {0}. Time taken for epoch: {1:.2f} seconds.\n'.format(optimizer.param_groups[0]['lr'], time.time() - t0))
            
            if args.wandb:
                wandb.log({'epoch':epoch, 'val acc': val_acc})
        print('Testing model')
        model.load_state_dict(torch.load(args.save))
        evaluate(args, model, device, test_loader, is_test_set=True)
        print_and_log("\nIteration end: {0}/{1}\n".format(i+1, args.iters))

        layer_fired_weights, total_fired_weights = mask.fired_masks_update()
        for name in layer_fired_weights:
            print('The final percentage of fired weights in the layer', name, 'is:', layer_fired_weights[name])
        print('The final percentage of the total fired weights is:', total_fired_weights)

        if args.wandb:
            wandb.finish()

if __name__ == '__main__':
   main()
