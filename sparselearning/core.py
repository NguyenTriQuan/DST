from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy

import numpy as np
import math
import wandb
from sparselearning.models import Residual

def add_sparse_args(parser):
    parser.add_argument('--sparse', action='store_true', help='Enable sparse mode. Default: True.')
    parser.add_argument('--fix', action='store_true', help='Fix sparse connectivity during training. Default: True.')
    parser.add_argument('--sparse_init', type=str, default='ERK', help='sparse initialization')
    parser.add_argument('--growth', type=str, default='random', help='Growth mode. Choose from: momentum, random, random_unfired, and gradient.')
    parser.add_argument('--death', type=str, default='magnitude', help='Death mode / pruning mode. Choose from: magnitude, SET, threshold.')
    parser.add_argument('--redistribution', type=str, default='none', help='Redistribution mode. Choose from: momentum, magnitude, nonzeros, or none.')
    parser.add_argument('--death-rate', type=float, default=0.50, help='The pruning rate / death rate.')
    parser.add_argument('--density', type=float, default=0.05, help='The density of the overall sparse network.')
    parser.add_argument('--update_frequency', type=int, default=100, metavar='N', help='how many iterations to train between parameter exploration')
    parser.add_argument('--decay-schedule', type=str, default='cosine', help='The decay schedule for the pruning rate. Default: cosine. Choose from: cosine, linear.')
class CosineDecay(object):
    def __init__(self, death_rate, T_max, eta_min=0.005, last_epoch=-1):
        self.sgd = optim.SGD(torch.nn.ParameterList([torch.nn.Parameter(torch.zeros(1))]), lr=death_rate)
        self.cosine_stepper = torch.optim.lr_scheduler.CosineAnnealingLR(self.sgd, T_max, eta_min, last_epoch)

    def step(self):
        self.cosine_stepper.step()

    def get_dr(self):
        return self.sgd.param_groups[0]['lr']

class LinearDecay(object):
    def __init__(self, death_rate, factor=0.99, frequency=600):
        self.factor = factor
        self.steps = 0
        self.frequency = frequency

    def step(self):
        self.steps += 1

    def get_dr(self, death_rate):
        if self.steps > 0 and self.steps % self.frequency == 0:
            return death_rate*self.factor
        else:
            return death_rate

class TopK(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scores, k):
        # Get the supermask by sorting the scores and using the top k%
        out = scores.clone()
        _, idx = scores.flatten().sort()
        # j = int((1 - k) * scores.numel())

        # flat_out and out access the same memory.
        flat_out = out.flatten()
        flat_out[idx[:k]] = 0
        flat_out[idx[k:]] = 1
        return out

    @staticmethod
    def backward(ctx, g):
        # send the gradient g straight-through on the backward pass.
        return g, None
    
class NotZeros(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scores):
        # Get the supermask by sorting the scores and using the top k%
        out = scores.clone()
        # _, idx = scores.flatten().sort()
        # j = int((1 - k) * scores.numel())

        # flat_out and out access the same memory.
        # flat_out = out.flatten()
        out[out == 0] = 0
        out[out != 0] = 1
        return out

    @staticmethod
    def backward(ctx, g):
        # send the gradient g straight-through on the backward pass.
        return g, None
    
# NPB core #

def measure_node_path(model, grad_dummy):
    grad_dummy = torch.autograd.grad(eff_paths, dummy, retain_graph=True, create_graph=True)
    eff_nodes = 0
    i = 0
    for m in model.modules():
        if hasattr(m, 'score'):
            # grad_dummy, = torch.autograd.grad(eff_paths, m.dummy, retain_graph=True, create_graph=True)
            if len(m.weight.shape) == 4:
                # temp = NotZeros.apply(m.weight.grad).sum((1,2,3))
                # eff_nodes += torch.clamp(temp, max=1).sum()
                eff_nodes += torch.sign(grad_dummy[i].abs().sum((1,2,3))).sum()
            else:
                # temp = NotZeros.apply(m.weight.grad).sum((1))
                # eff_nodes += torch.clamp(temp, max=1).sum()
                # print(m.weight.grad.abs().max())
                eff_nodes += torch.sign(grad_dummy[i].abs().sum((1))).sum()
            # eff_nodes += m.eff_nodes
            # m.eff_nodes = None
            i += 1
        m.measure = False
    # print(eff_nodes, eff_paths)
    # eff_nodes = eff_nodes.log()
    return eff_nodes, eff_paths

def NPB_linear_forward(self, x):
    # if self.measure:
    #     # nodes_in, paths_in = x

    #     # if paths_in.shape[0] != self.mask.shape[1]:
    #     #     s = math.ceil(math.sqrt(self.mask.shape[1]/paths_in.shape[0]))
    #     #     paths_in = paths_in.view(paths_in.shape[0], 1, 1).expand(paths_in.shape[0], s, s).contiguous().view(-1)
    #     #     nodes_in = nodes_in.view(nodes_in.shape[0], 1, 1).expand(nodes_in.shape[0], s, s).contiguous().view(-1)

    #     # paths_out = torch.logsumexp(torch.log(self.mask+1e-12) + paths_in.view((1,-1)), dim=(1))
    #     # nodes_in = torch.clamp(torch.sum(self.mask, dim=(0)) * nodes_in, max=1)
    #     # self.eff_nodes = nodes_in.sum()
    #     # nodes_out = torch.clamp(torch.sum(self.mask * nodes_in.view((1,-1)), dim=(1)), max=1)
    #     # return nodes_out, paths_out

    #     x_max = torch.max(x)
    #     return torch.log(F.linear((x-x_max).exp(), self.mask*self.dummy, None)+1e-6) + x_max
    # else:
    if self.training:
        self.mask = TopK.apply(self.score.abs(), self.num_zeros)
        eff_paths, images = x
        max_paths = eff_paths.max()
        eff_paths = torch.log(F.linear((eff_paths - max_paths).exp(), self.mask * self.dummy, None)+1e-6) + max_paths
        out = F.linear(images, self.mask * self.weight, self.bias)
        return eff_paths, out
    else:
        return F.linear(x, self.mask * self.weight, self.bias)
    
def NPB_conv_forward(self, x):
    # if self.measure:
    #     # nodes_in, paths_in = x
    #     # paths_out = torch.logsumexp(torch.log(self.mask+1e-12) + paths_in.view((1,-1,1,1)), dim=(1,2,3))
    #     # nodes_in = torch.clamp(torch.sum(self.mask, dim=(0,2,3)) * nodes_in, max=1)
    #     # self.eff_nodes = nodes_in.sum()
    #     # nodes_out = torch.clamp(torch.sum(self.mask * nodes_in.view((1,-1,1,1)), dim=(1,2,3)), max=1)
    #     # return nodes_out, paths_out
    #     x_max = torch.max(x)
    #     return torch.log(self._conv_forward((x-x_max).exp(), self.mask*self.dummy, None)+1e-6) + x_max
    # else:
    if self.training:
        self.mask = TopK.apply(self.score.abs(), self.num_zeros)
        eff_paths, images = x
        max_paths = eff_paths.max()
        eff_paths = torch.log(self._conv_forward((eff_paths - max_paths).exp(), self.mask * self.dummy, None)+1e-6) + max_paths
        out = self._conv_forward(images, self.weight * self.mask, self.bias)
        return eff_paths, out
    else:
        return self._conv_forward(x, self.weight * self.mask, self.bias)
    
def NPB_dummy_forward(self, x):
    if self.training:
        return x[0], self.original_forward(x[1])
    else:
        return self.original_forward(x)
    
def NPB_stable_forward(self, x):
    if self.training:
        eff_paths, images = x
        max_paths = eff_paths.max()
        eff_paths = torch.log(self.original_forward((eff_paths - max_paths).exp())+1e-6) + max_paths
        out = self.original_forward(images)
        return eff_paths, out
    else:
        return self.original_forward(x)

def NPB_residual_forward(self, x, y):
    if self.training:
        # nodes_in_x, paths_in_x = x
        # nodes_in_y, paths_in_y = y
        # nodes_out = torch.maximum(nodes_in_x, nodes_in_y)
        # paths_out = torch.logsumexp(torch.stack([paths_in_x, paths_in_y], dim=0), dim=0)
        # return nodes_out, paths_out
        # print((x==-math.inf).sum(), (x==-math.inf).sum())
        return torch.logsumexp(torch.stack([x[0], y[0]], dim=0), dim=0), x[1] + y[1]
    else:
        return self.original_forward(x, y)

def NPB_register(model):
    # model.apply(lambda m: setattr(m, "measure", False))
    for m in model.modules():
        if isinstance(m, nn.Linear):
            m.score = nn.Parameter(torch.empty_like(m.weight), requires_grad=True).cuda()
            m.dummy = torch.ones_like(m.weight, requires_grad=True).cuda()
            nn.init.kaiming_normal_(m.score)
            setattr(m, 'original_forward', m.forward)
            setattr(m, 'forward', NPB_linear_forward.__get__(m, m.__class__))
        elif isinstance(m, nn.Conv2d):
            m.score = nn.Parameter(torch.empty_like(m.weight), requires_grad=True).cuda()
            m.dummy = torch.ones_like(m.weight, requires_grad=True).cuda()
            nn.init.kaiming_normal_(m.score)
            setattr(m, 'original_forward', m.forward)
            setattr(m, 'forward', NPB_conv_forward.__get__(m, m.__class__))
        elif isinstance(m, Residual):
            setattr(m, 'original_forward', m.forward)
            setattr(m, 'forward', NPB_residual_forward.__get__(m, m.__class__))
        elif isinstance(m, nn.MaxPool2d) or isinstance(m, nn.AvgPool2d) or isinstance(m, nn.Flatten):
            setattr(m, 'original_forward', m.forward)
            setattr(m, 'forward', NPB_stable_forward.__get__(m, m.__class__))
        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.LogSoftmax) or isinstance(m, nn.ReLU) or isinstance(m, nn.Dropout):
            setattr(m, 'original_forward', m.forward)
            setattr(m, 'forward', NPB_dummy_forward.__get__(m, m.__class__))
    
    model.dummy = [m.dummy for m in model.modules() if hasattr(m, 'dummy')]
    model.weights = [m.weight for m in model.modules() if hasattr(m, 'dummy')]

            
class Masking(object):
    def __init__(self, optimizer, death_rate=0.3, growth_death_ratio=1.0, death_rate_decay=None, death_mode='magnitude', growth_mode='momentum', redistribution_mode='momentum', threshold=0.001, args=None):
        growth_modes = ['random', 'momentum', 'momentum_neuron', 'gradient']
        if growth_mode not in growth_modes:
            print('Growth mode: {0} not supported!'.format(growth_mode))
            print('Supported modes are:', str(growth_modes))

        self.args = args
        self.device = torch.device("cuda")
        self.growth_mode = growth_mode
        self.death_mode = death_mode
        self.growth_death_ratio = growth_death_ratio
        self.redistribution_mode = redistribution_mode
        self.death_rate_decay = death_rate_decay

        self.masks = {}
        self.scores = {}
        self.modules = []
        self.names = []
        self.optimizer = optimizer

        # stats
        self.name2zeros = {}
        self.num_remove = {}
        self.name2nonzeros = {}
        self.death_rate = death_rate
        self.baseline_nonzero = None
        self.steps = 0

        # if fix, then we do not explore the sparse connectivity
        if self.args.fix: self.prune_every_k_steps = None
        else: self.prune_every_k_steps = self.args.update_frequency

    def init(self, mode='ERK', density=0.05, erk_power_scale=1.0):
        self.density = density
        if mode == 'GMP':
            self.baseline_nonzero = 0
            for module in self.modules:
                for name, weight in module.named_parameters():
                    if name not in self.masks: continue
                    self.masks[name] = torch.ones_like(weight, dtype=torch.float32, requires_grad=False).cuda()
                    self.baseline_nonzero += (self.masks[name] != 0).sum().int().item()

        elif mode == 'lottery_ticket':
            print('initialize by lottery ticket')
            self.baseline_nonzero = 0
            weight_abs = []
            for module in self.modules:
                for name, weight in module.named_parameters():
                    if name not in self.masks: continue
                    weight_abs.append(torch.abs(weight))

            # Gather all scores in a single vector and normalise
            all_scores = torch.cat([torch.flatten(x) for x in weight_abs])
            num_params_to_keep = int(len(all_scores) * self.density)

            threshold, _ = torch.topk(all_scores, num_params_to_keep, sorted=True)
            acceptable_score = threshold[-1]

            for module in self.modules:
                for name, weight in module.named_parameters():
                    if name not in self.masks: continue
                    self.masks[name] = ((torch.abs(weight)) >= acceptable_score).float()
                    self.baseline_nonzero += (self.masks[name]!=0).sum().int().item()

        elif mode == 'uniform':
            self.baseline_nonzero = 0
            for module in self.modules:
                for name, weight in module.named_parameters():
                    if name not in self.masks: continue
                    self.masks[name][:] = (torch.rand(weight.shape) < density).float().data.cuda() #lsw
                    # self.masks[name][:] = (torch.rand(weight.shape) < density).float().data #lsw
                    self.baseline_nonzero += weight.numel()*density

        elif mode == 'ERK':
            print('initialize by ERK')
            total_params = 0
            for name, weight in self.masks.items():
                total_params += weight.numel()
            is_epsilon_valid = False
            # # The following loop will terminate worst case when all masks are in the
            # custom_sparsity_map. This should probably never happen though, since once
            # we have a single variable or more with the same constant, we have a valid
            # epsilon. Note that for each iteration we add at least one variable to the
            # custom_sparsity_map and therefore this while loop should terminate.
            dense_layers = set()
            while not is_epsilon_valid:
                # We will start with all layers and try to find right epsilon. However if
                # any probablity exceeds 1, we will make that layer dense and repeat the
                # process (finding epsilon) with the non-dense layers.
                # We want the total number of connections to be the same. Let say we have
                # for layers with N_1, ..., N_4 parameters each. Let say after some
                # iterations probability of some dense layers (3, 4) exceeded 1 and
                # therefore we added them to the dense_layers set. Those layers will not
                # scale with erdos_renyi, however we need to count them so that target
                # paratemeter count is achieved. See below.
                # eps * (p_1 * N_1 + p_2 * N_2) + (N_3 + N_4) =
                #    (1 - default_sparsity) * (N_1 + N_2 + N_3 + N_4)
                # eps * (p_1 * N_1 + p_2 * N_2) =
                #    (1 - default_sparsity) * (N_1 + N_2) - default_sparsity * (N_3 + N_4)
                # eps = rhs / (\sum_i p_i * N_i) = rhs / divisor.

                divisor = 0
                rhs = 0
                raw_probabilities = {}
                for name, mask in self.masks.items():
                    n_param = np.prod(mask.shape)
                    n_zeros = n_param * (1 - self.density)
                    n_ones = n_param * self.density

                    if name in dense_layers:
                        rhs -= n_zeros
                    else:
                        rhs += n_ones
                        raw_probabilities[name] = (
                                                          np.sum(mask.shape) / np.prod(mask.shape)
                                                  ) ** erk_power_scale
                        divisor += raw_probabilities[name] * n_param
                epsilon = rhs / divisor
                max_prob = np.max(list(raw_probabilities.values()))
                max_prob_one = max_prob * epsilon
                if max_prob_one > 1:
                    is_epsilon_valid = False
                    for mask_name, mask_raw_prob in raw_probabilities.items():
                        if mask_raw_prob == max_prob:
                            print(f"Sparsity of var:{mask_name} had to be set to 0.")
                            dense_layers.add(mask_name)
                else:
                    is_epsilon_valid = True

            density_dict = {}
            total_nonzero = 0.0
            # With the valid epsilon, we can set sparsities of the remaning layers.
            for name, mask in self.masks.items():
                n_param = np.prod(mask.shape)
                if name in dense_layers:
                    density_dict[name] = 1.0
                else:
                    probability_one = epsilon * raw_probabilities[name]
                    density_dict[name] = probability_one
                print(
                    f"layer: {name}, shape: {mask.shape}, density: {density_dict[name]}"
                )
                self.masks[name][:] = (torch.rand(mask.shape) < density_dict[name]).float().data.cuda()
                mask = self.masks[name]
                self.name2nonzeros[name] = mask.sum().item()
                self.name2zeros[name] = mask.numel() - self.name2nonzeros[name]
                total_nonzero += density_dict[name] * mask.numel()
                for n, m in self.modules[-1].named_modules():
                    if hasattr(m, 'score'):
                        if n in name:
                            m.num_zeros = math.ceil((1-density_dict[name])*m.weight.numel())
                            m.mask = TopK.apply(m.score.abs(), m.num_zeros).detach()
                            m.fired_mask = m.mask.clone()
            print(f"Overall sparsity {total_nonzero / total_params}")

        self.apply_mask()
        self.fired_masks = copy.deepcopy(self.masks) # used for ITOP
        # self.print_nonzero_counts()

        total_size = 0
        for name, weight in self.masks.items():
            total_size  += weight.numel()
        print('Total Model parameters:', total_size)

        sparse_size = 0
        for name, weight in self.masks.items():
            sparse_size += (weight != 0).sum().int().item()

        print('Total parameters under sparsity level of {0}: {1}'.format(self.density, sparse_size / total_size))


    def step(self):
        # self.optimizer.step()
        if self.args.method == 'score_npb':
            pass
        else:
            self.apply_mask()
        self.death_rate_decay.step()
        self.death_rate = self.death_rate_decay.get_dr()
        self.steps += 1

        if self.prune_every_k_steps is not None:
            if self.steps % self.prune_every_k_steps == 0:
                if self.args.method == 'score_npb':
                    self.fired_masks_update()
                else:
                    self.truncate_weights()
                    _, _ = self.fired_masks_update()
                    self.print_nonzero_counts()


    def add_module(self, module, density, sparse_init='ER'):
        self.modules.append(module)
        for name, tensor in module.named_parameters():
            self.names.append(name)
            self.masks[name] = torch.zeros_like(tensor, dtype=torch.float32, requires_grad=False).cuda()
            # self.scores[name] = nn.Parameter(torch.empty_like(tensor), requires_grad=True).cuda() 
            # nn.init.kaiming_normal_(self.scores[name])
            # for n, m in module.named_modules():
            #     if n in name:
            #         m.name = name


        print('Removing biases...')
        self.remove_weight_partial_name('bias')
        print('Removing 2D batch norms...')
        self.remove_type(nn.BatchNorm2d)
        print('Removing 1D batch norms...')
        self.remove_type(nn.BatchNorm1d)
        
        if self.args.method == 'score_npb':
            NPB_register(module)

        self.init(mode=sparse_init, density=density)

    def remove_weight(self, name):
        if name in self.masks:
            print('Removing {0} of size {1} = {2} parameters.'.format(name, self.masks[name].shape,
                                                                      self.masks[name].numel()))
            self.masks.pop(name)
        elif name + '.weight' in self.masks:
            print('Removing {0} of size {1} = {2} parameters.'.format(name, self.masks[name + '.weight'].shape,
                                                                      self.masks[name + '.weight'].numel()))
            self.masks.pop(name + '.weight')
        else:
            print('ERROR', name)

    def remove_weight_partial_name(self, partial_name):
        removed = set()
        for name in list(self.masks.keys()):
            if partial_name in name:

                print('Removing {0} of size {1} with {2} parameters...'.format(name, self.masks[name].shape,
                                                                                   np.prod(self.masks[name].shape)))
                removed.add(name)
                self.masks.pop(name)

        print('Removed {0} layers.'.format(len(removed)))

        i = 0
        while i < len(self.names):
            name = self.names[i]
            if name in removed:
                self.names.pop(i)
            else:
                i += 1

    def remove_type(self, nn_type):
        for module in self.modules:
            for name, module in module.named_modules():
                if isinstance(module, nn_type):
                    self.remove_weight(name)

    def apply_mask(self):
        if self.args.method == 'score_npb': return
        for module in self.modules:
            for name, tensor in module.named_parameters():
                if name in self.masks:
                    tensor.data = tensor.data*self.masks[name]
                    # reset momentum
                    if 'momentum_buffer' in self.optimizer.state[tensor]:
                        self.optimizer.state[tensor]['momentum_buffer'] = self.optimizer.state[tensor]['momentum_buffer']*self.masks[name]

    def truncate_weights_GMP(self, epoch):
        '''
        Implementation  of GMP To prune, or not to prune: exploring the efficacy of pruning for model compression https://arxiv.org/abs/1710.01878
        :param epoch: current training epoch
        :return:
        '''
        prune_rate = 1 - self.density
        curr_prune_epoch = epoch
        total_prune_epochs = self.args.multiplier * self.args.final_prune_epoch - self.args.multiplier * self.args.init_prune_epoch + 1
        if epoch >= self.args.multiplier * self.args.init_prune_epoch and epoch <= self.args.multiplier * self.args.final_prune_epoch:
            prune_decay = (1 - ((curr_prune_epoch - self.args.multiplier * self.args.init_prune_epoch) / total_prune_epochs)) ** 3
            curr_prune_rate = prune_rate - (prune_rate * prune_decay)

            for module in self.modules:
                for name, weight in module.named_parameters():
                    if name not in self.masks: continue

                    x, idx = torch.sort(torch.abs(weight.data.view(-1)))
                    p = int(curr_prune_rate * weight.numel())
                    self.masks[name].data.view(-1)[idx[:p]] = 0.0
            self.apply_mask()
        total_size = 0
        for name, weight in self.masks.items():
            total_size += weight.numel()
        print('Total Model parameters:', total_size)

        sparse_size = 0
        for name, weight in self.masks.items():
            sparse_size += (weight != 0).sum().int().item()

        print('Total parameters under sparsity level of {0}: {1} after epoch of {2}'.format(self.density, sparse_size / total_size, epoch))

    def truncate_weights(self):


        for module in self.modules:
            for name, weight in module.named_parameters():
                if name not in self.masks: continue
                mask = self.masks[name]
                self.name2nonzeros[name] = mask.sum().item()
                self.name2zeros[name] = mask.numel() - self.name2nonzeros[name]

                # death
                if self.death_mode == 'magnitude':
                    new_mask = self.magnitude_death(mask, weight, name)
                elif self.death_mode == 'SET':
                    new_mask = self.magnitude_and_negativity_death(mask, weight, name)
                elif self.death_mode == 'Taylor_FO':
                    new_mask = self.taylor_FO(mask, weight, name)
                elif self.death_mode == 'threshold':
                    new_mask = self.threshold_death(mask, weight, name)

                self.num_remove[name] = int(self.name2nonzeros[name] - new_mask.sum().item())
                self.masks[name][:] = new_mask


        for module in self.modules:
            for name, weight in module.named_parameters():
                if name not in self.masks: continue
                new_mask = self.masks[name].data.byte()

                # growth
                if self.growth_mode == 'random':
                    new_mask = self.random_growth(name, new_mask, weight)

                if self.growth_mode == 'random_unfired':
                    new_mask = self.random_unfired_growth(name, new_mask, weight)

                elif self.growth_mode == 'momentum':
                    new_mask = self.momentum_growth(name, new_mask, weight)

                elif self.growth_mode == 'gradient':
                    new_mask = self.gradient_growth(name, new_mask, weight)

                new_nonzero = new_mask.sum().item()

                # exchanging masks
                self.masks.pop(name)
                self.masks[name] = new_mask.float()

        self.apply_mask()

    '''
                    NPB
    '''
    def score_NPB_reg(self):
        eff_nodes = 0
        P_out = torch.zeros(3).float().cuda()
        eff_nodes_out = torch.ones(3).float().cuda()
        for m in self.modules[-1].modules():
            if not hasattr(m, 'score'): continue
            mask = m.mask
            if len(mask.shape) == 4:
                dim_in = (0,2,3)
                dim_out = (1,2,3)
                view_in = (1,-1,1,1)
            else:
                dim_in = (0)
                dim_out = (1)
                view_in = (1,-1)

            if P_out.shape[0] != mask.shape[1]:
                s = math.ceil(math.sqrt(mask.shape[1]/P_out.shape[0]))
                P_out = P_out.view(P_out.shape[0], 1, 1).expand(P_out.shape[0], s, s).contiguous().view(-1)
                eff_nodes_out = eff_nodes_out.view(eff_nodes_out.shape[0], 1, 1).expand(eff_nodes_out.shape[0], s, s).contiguous().view(-1)

            P_out = torch.logsumexp(torch.log(mask+1e-12) + P_out.view(view_in), dim=dim_out)
            eff_nodes_in = torch.clamp(torch.sum(mask, dim=dim_in) * eff_nodes_out, max=1)
            eff_nodes += eff_nodes_in.sum()
            eff_nodes_out = torch.clamp(torch.sum(mask * eff_nodes_out.view(view_in), dim=dim_out), max=1)
        eff_nodes += eff_nodes_out.sum()
        P_out = torch.logsumexp(P_out, dim=0)
        if self.steps % self.prune_every_k_steps == 0:
            print(f'eff nodes: {eff_nodes}, eff paths: {P_out}')
            if self.args.wandb:
                wandb.log({'eff nodes': eff_nodes, 'eff paths': P_out})
        return self.args.alpha * eff_nodes.log() + (1-self.args.alpha) * P_out
    
    def NPB_reg(self):
        for module in self.modules:
            eff_nodes = 0
            P_out = torch.zeros(3).float().cuda()
            eff_nodes_out = torch.ones(3).float().cuda()
            for name, tensor in module.named_parameters():
                if name not in self.masks: continue

                # num_remove = math.ceil(self.death_rate*self.name2nonzeros[name])
                num_zeros = self.name2zeros[name]
                # k = math.ceil(num_zeros + num_remove)
                k = math.ceil(num_zeros)
                mask = TopK.apply(tensor.abs(), k)

                if len(tensor.shape) == 4:
                    dim_in = (0,2,3)
                    dim_out = (1,2,3)
                    view_in = (1,-1,1,1)
                else:
                    dim_in = (0)
                    dim_out = (1)
                    view_in = (1,-1)

                if P_out.shape[0] != mask.shape[1]:
                    s = math.ceil(math.sqrt(mask.shape[1]/P_out.shape[0]))
                    P_out = P_out.view(P_out.shape[0], 1, 1).expand(P_out.shape[0], s, s).contiguous().view(-1)
                    eff_nodes_out = eff_nodes_out.view(eff_nodes_out.shape[0], 1, 1).expand(eff_nodes_out.shape[0], s, s).contiguous().view(-1)

                P_out = torch.logsumexp(torch.log(mask+1e-12) + P_out.view(view_in), dim=dim_out)
                eff_nodes_in = torch.clamp(torch.sum(mask, dim=dim_in) * eff_nodes_out, max=1)
                eff_nodes += eff_nodes_in.sum()
                eff_nodes_out = torch.clamp(torch.sum(mask * eff_nodes_out.view(view_in), dim=dim_out), max=1)
            eff_nodes += eff_nodes_out.sum()
            P_out = torch.logsumexp(P_out, dim=0)
            if self.steps % self.prune_every_k_steps == 0:
                print(f'eff nodes: {eff_nodes}, eff paths: {P_out}')
                if self.args.wandb:
                    wandb.log({'eff nodes': eff_nodes, 'eff paths': P_out})
        return self.args.alpha * eff_nodes.log() + (1-self.args.alpha) * P_out

    '''
                    DEATH
    '''

    def threshold_death(self, mask, weight, name):
        return (torch.abs(weight.data) > self.threshold)

    def taylor_FO(self, mask, weight, name):

        num_remove = math.ceil(self.death_rate * self.name2nonzeros[name])
        num_zeros = self.name2zeros[name]
        k = math.ceil(num_zeros + num_remove)

        x, idx = torch.sort((weight.data * weight.grad).pow(2).flatten())
        mask.data.view(-1)[idx[:k]] = 0.0

        return mask

    def magnitude_death(self, mask, weight, name):

        num_remove = math.ceil(self.death_rate*self.name2nonzeros[name])
        if num_remove == 0.0: return weight.data != 0.0
        num_zeros = self.name2zeros[name]

        x, idx = torch.sort(torch.abs(weight.data.view(-1)))
        n = idx.shape[0]

        k = math.ceil(num_zeros + num_remove)
        threshold = x[k-1].item()

        return (torch.abs(weight.data) > threshold)


    def magnitude_and_negativity_death(self, mask, weight, name):
        num_remove = math.ceil(self.death_rate*self.name2nonzeros[name])
        num_zeros = self.name2zeros[name]

        # find magnitude threshold
        # remove all weights which absolute value is smaller than threshold
        x, idx = torch.sort(weight[weight > 0.0].data.view(-1))
        k = math.ceil(num_remove/2.0)
        if k >= x.shape[0]:
            k = x.shape[0]

        threshold_magnitude = x[k-1].item()

        # find negativity threshold
        # remove all weights which are smaller than threshold
        x, idx = torch.sort(weight[weight < 0.0].view(-1))
        k = math.ceil(num_remove/2.0)
        if k >= x.shape[0]:
            k = x.shape[0]
        threshold_negativity = x[k-1].item()


        pos_mask = (weight.data > threshold_magnitude) & (weight.data > 0.0)
        neg_mask = (weight.data < threshold_negativity) & (weight.data < 0.0)


        new_mask = pos_mask | neg_mask
        return new_mask

    '''
                    GROWTH
    '''

    def random_unfired_growth(self, name, new_mask, weight):
        total_regrowth = self.num_remove[name]
        n = (new_mask == 0).sum().item()
        if n == 0: return new_mask
        num_nonfired_weights = (self.fired_masks[name]==0).sum().item()

        if total_regrowth <= num_nonfired_weights:
            idx = (self.fired_masks[name].flatten() == 0).nonzero()
            indices = torch.randperm(len(idx))[:total_regrowth]

            # idx = torch.nonzero(self.fired_masks[name].flatten())
            new_mask.data.view(-1)[idx[indices]] = 1.0
        else:
            new_mask[self.fired_masks[name]==0] = 1.0
            n = (new_mask == 0).sum().item()
            expeced_growth_probability = ((total_regrowth-num_nonfired_weights) / n)
            new_weights = torch.rand(new_mask.shape).cuda() < expeced_growth_probability
            new_mask = new_mask.byte() | new_weights
        return new_mask

    def random_growth(self, name, new_mask, weight):
        total_regrowth = self.num_remove[name]
        n = (new_mask==0).sum().item()
        if n == 0: return new_mask
        expeced_growth_probability = (total_regrowth/n)
        new_weights = torch.rand(new_mask.shape).cuda() < expeced_growth_probability
        new_mask_ = new_mask.byte() | new_weights
        if (new_mask_!=0).sum().item() == 0:
            new_mask_ = new_mask
        return new_mask_

    def momentum_growth(self, name, new_mask, weight):
        total_regrowth = self.num_remove[name]
        grad = self.get_momentum_for_weight(weight)
        grad = grad*(new_mask==0).float()
        y, idx = torch.sort(torch.abs(grad).flatten(), descending=True)
        new_mask.data.view(-1)[idx[:total_regrowth]] = 1.0

        return new_mask

    def gradient_growth(self, name, new_mask, weight):
        total_regrowth = self.num_remove[name]
        grad = self.get_gradient_for_weights(weight)
        grad = grad*(new_mask==0).float()

        y, idx = torch.sort(torch.abs(grad).flatten(), descending=True)
        new_mask.data.view(-1)[idx[:total_regrowth]] = 1.0

        return new_mask



    def momentum_neuron_growth(self, name, new_mask, weight):
        total_regrowth = self.num_remove[name]
        grad = self.get_momentum_for_weight(weight)

        M = torch.abs(grad)
        if len(M.shape) == 2: sum_dim = [1]
        elif len(M.shape) == 4: sum_dim = [1, 2, 3]

        v = M.mean(sum_dim).data
        v /= v.sum()

        slots_per_neuron = (new_mask==0).sum(sum_dim)

        M = M*(new_mask==0).float()
        for i, fraction  in enumerate(v):
            neuron_regrowth = math.floor(fraction.item()*total_regrowth)
            available = slots_per_neuron[i].item()

            y, idx = torch.sort(M[i].flatten())
            if neuron_regrowth > available:
                neuron_regrowth = available
            threshold = y[-(neuron_regrowth)].item()
            if threshold == 0.0: continue
            if neuron_regrowth < 10: continue
            new_mask[i] = new_mask[i] | (M[i] > threshold)

        return new_mask

    '''
                UTILITY
    '''
    def get_momentum_for_weight(self, weight):
        if 'exp_avg' in self.optimizer.state[weight]:
            adam_m1 = self.optimizer.state[weight]['exp_avg']
            adam_m2 = self.optimizer.state[weight]['exp_avg_sq']
            grad = adam_m1/(torch.sqrt(adam_m2) + 1e-08)
        elif 'momentum_buffer' in self.optimizer.state[weight]:
            grad = self.optimizer.state[weight]['momentum_buffer']
        return grad

    def get_gradient_for_weights(self, weight):
        grad = weight.grad.clone()
        return grad

    def print_nonzero_counts(self):
        for module in self.modules:
            for name, tensor in module.named_parameters():
                if name not in self.masks: continue
                mask = self.masks[name]
                num_nonzeros = (mask != 0).sum().item()
                val = '{0}: {1}->{2}, density: {3:.3f}'.format(name, self.name2nonzeros[name], num_nonzeros, num_nonzeros/float(mask.numel()))
                print(val)


        for module in self.modules:
            for name, tensor in module.named_parameters():
                if name not in self.masks: continue
                print('Death rate: {0}\n'.format(self.death_rate))
                break

    def fired_masks_update(self):
        ntotal_fired_weights = 0.0
        ntotal_weights = 0.0
        layer_fired_weights = {}
        if self.args.method == 'score_npb':
            for module in self.modules:
                for m in module.modules():
                    if not hasattr(m, 'score'): continue
                    m.fired_mask = m.mask.data.byte() | m.fired_mask.data.byte()
                    ntotal_fired_weights += float(m.fired_mask.sum().item())
                    ntotal_weights += float(m.fired_mask.numel())
            total_fired_weights = ntotal_fired_weights/ntotal_weights
            print('The percentage of the total fired weights is:', total_fired_weights)
        else:
            for module in self.modules:
                for name, weight in module.named_parameters():
                    if name not in self.masks: continue
                    self.fired_masks[name] = self.masks[name].data.byte() | self.fired_masks[name].data.byte()
                    ntotal_fired_weights += float(self.fired_masks[name].sum().item())
                    ntotal_weights += float(self.fired_masks[name].numel())
                    layer_fired_weights[name] = float(self.fired_masks[name].sum().item())/float(self.fired_masks[name].numel())
                    print('Layerwise percentage of the fired weights of', name, 'is:', layer_fired_weights[name])
            total_fired_weights = ntotal_fired_weights/ntotal_weights
            print('The percentage of the total fired weights is:', total_fired_weights)
        if self.args.wandb:
            wandb.log({'fired weights': total_fired_weights})
        return layer_fired_weights, total_fired_weights
    
import random
def get_related_layers(net, input_shape):
    def stable_check(x1, x2):
        eps = x1 * 1e-6
        return abs(x1 - x2) < eps

    def forward_hook(m, i, o):
        m.output_idx = random.random() * m.input_idx
        o[0] = o[0]*0 + m.output_idx
        return
    
    def forward_pre_hook(m, i):
        if len(i[0].shape) == 4:
            m.input_idx = i[0][0,0,0,0].item()
        else:
            m.input_idx = i[0][0,0].item()
        return (torch.zeros_like(i[0]))
    
    def forward_bn_hook(m, i, o):
        o[0] = i[0]
        return

    handes = []
    for m in net.modules():
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            h1 = m.register_forward_pre_hook(forward_pre_hook)
            h2 = m.register_forward_hook(forward_hook)
            handes += [h1, h2]
        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            h3 = m.register_forward_hook(forward_bn_hook)
            handes += [h3]

    c, h, w = input_shape
    data = torch.ones((1, c, h, w), dtype=torch.float, device='cuda')
    net.eval()
    out = net(data)  
    for h in handes:
        h.remove()
    
    scores = [m for m in net.modules() if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d)]
    for n, m in net.named_modules():
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            # print(n, m.weight.shape, m.input_idx, m.output_idx)
            m.name = n
            m.prev_layers = []
            m.next_layers = []
    eps = 1e-3
    for i in range(len(scores)):
        for j in range(i):
            # if abs(scores[i].input_idx - scores[j].output_idx) < eps:
            if stable_check(scores[i].input_idx, scores[j].output_idx):
                scores[i].prev_layers.append(scores[j])
                scores[j].next_layers.append(scores[i])
            else:
                for k in range(j):
                    # if abs(scores[i].input_idx - scores[j].output_idx - scores[k].output_idx) < eps:
                    if stable_check(scores[i].input_idx, scores[j].output_idx + scores[k].output_idx):
                        scores[i].prev_layers += [scores[j], scores[k]]
                        scores[j].next_layers += [scores[i]]
                        scores[k].next_layers += [scores[i]]

                    # if abs(scores[i].input_idx - scores[j].output_idx - scores[k].input_idx) < eps:
                    if stable_check(scores[i].input_idx, scores[j].output_idx + scores[k].input_idx):
                        scores[i].prev_layers += [scores[j]] + scores[k].prev_layers
                        scores[j].next_layers += [scores[i]]
                        for g in scores[k].prev_layers:
                            g.next_layers += [scores[i]]

                    # if abs(scores[i].input_idx - scores[k].output_idx - scores[j].input_idx) < eps:
                    if stable_check(scores[i].input_idx, scores[k].output_idx + scores[j].input_idx):
                        scores[i].prev_layers += [scores[k]] + scores[j].prev_layers
                        scores[k].next_layers += [scores[i]]
                        for g in scores[j].prev_layers:
                            g.next_layers += [scores[i]]

                    # if abs(scores[i].input_idx - scores[j].input_idx - scores[k].input_idx) < eps:
                    if stable_check(scores[i].input_idx, scores[j].input_idx + scores[k].input_idx):
                        scores[i].prev_layers += scores[j].prev_layers + scores[k].prev_layers
                        for g in scores[j].prev_layers:
                            g.next_layers += [scores[i]]
                        for g in scores[k].prev_layers:
                            g.next_layers += [scores[i]]

    net.prev_layers = []
    for m in scores:
        m.prev_layers = tuple(set(m.prev_layers))
        m.next_layers = tuple(set(m.next_layers))
        if len(m.prev_layers) != 0:
            net.prev_layers.append(m.prev_layers)
        print(m.name, m.weight.shape, m.input_idx, m.output_idx)
        for j, n in enumerate(m.prev_layers):
            print('prev', j, n.name, n.weight.shape, n.input_idx, n.output_idx)
        for j, n in enumerate(m.next_layers):
            print('next', j, n.name, n.weight.shape, n.input_idx, n.output_idx)
        print()
    
    net.prev_layers = list(set(net.prev_layers))
    for i, layers in enumerate(net.prev_layers):
        if len(layers) == 0:
            print(i, layers)
        for m in layers:
            print(i, m.name)

    all_layers = []
    for layers in net.prev_layers:
        all_layers += list(layers)
    for m in scores[:-1]:
        assert m in all_layers