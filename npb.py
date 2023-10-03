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


class TopK(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scores, k):
        # Get the supermask by sorting the scores and using the top k%
        out = scores.clone()
        _, idx = scores.flatten().sort()
        # j = int((1 - k) * scores.numel())

        # flat_out and out access the same memory.
        flat_out = out.flatten()
        flat_out[idx[:k]] = 0.0
        flat_out[idx[k:]] = 1.0
        return out
    
    @staticmethod
    def backward(ctx, g):
        # send the gradient g straight-through on the backward pass.
        return g, None
    

def NPB_objective(model, args):
    model.zero_grad()
    ones = torch.ones((1, 3, 32, 32)).float().cuda()
    data = (0, ones)
    cum_max_paths, eff_paths = model(data)
    eff_paths = eff_paths.sum().log() + cum_max_paths
    eff_nodes = 0
    eff_paths.backward()
    for m in model.NPB_modules:
        layer_eff_nodes = (m.score.grad.abs() * m.mask).sum(m.dim_out)
        layer_eff_nodes = layer_eff_nodes > 0
        eff_nodes += layer_eff_nodes.sum()
        m.score.data = m.score.data + args.lr_score * m.score.grad.data * (args.alpha * (1-layer_eff_nodes.view(m.view_out)) + args.beta)

    return eff_paths, eff_nodes


def post_update(model):
    for m in model.NPB_modules:
        m.mask = m.get_mask().detach().clone()
        m.eff_paths = None
        # m.score.data = m.g * m.score.data / m.score.data.norm(2).detach()

def get_mask_by_weight(self):
    return TopK.apply(self.weight.abs(), self.num_zeros)

def get_mask_by_score(self):
    return TopK.apply(self.score.abs(), self.num_zeros)

def get_weight(self):
    return self.weight

def get_masked_weight(self):
    return self.mask * self.weight

def linear_forward(self, x, weight, bias):
    return F.linear(x, weight, bias)

def NPB_forward(self, x):
    
    if self.npb:
        cum_max_paths, eff_paths = x
        max_paths = eff_paths.max()
        self.mask = self.get_mask()
        eff_paths = self.base_func(eff_paths / max_paths, self.mask, None)
        return cum_max_paths + max_paths.log(), eff_paths
    else:
        return self.base_func(x, self.get_weight(), self.bias)
    
def NPB_dummy_forward(self, x):

    if self.npb:
        return x[0], x[1]
    else:
        return self.original_forward(x)
    
def NPB_stable_forward(self, x):

    if self.npb:
        return x[0], self.original_forward(x[1])
    else:
        return self.original_forward(x)

def NPB_residual_forward(self, x, y):

    if self.npb:
        if x[0] > y[0]:
            return x[0], x[1] + (y[1] / (x[0]-y[0]).exp())
        else:
            return y[0], (x[1] / (y[0]-x[0]).exp()) + y[1]
    else:
        return self.original_forward(x, y)

def NPB_register(model, args):
    model.apply(lambda m: setattr(m, "npb", False))
    NPB_modules = []
    for m in model.modules():
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            NPB_modules.append(m)
            setattr(m, 'original_forward', m.forward)
            setattr(m, 'post_update', post_update.__get__(m, m.__class__))
            m.num_zeros = 0

            m.score = nn.Parameter(torch.empty_like(m.weight), requires_grad=True).cuda()
            # nn.init.kaiming_normal_(m.score)
            nn.init.constant_(m.score, 0)
            setattr(m, 'get_weight', get_masked_weight.__get__(m, m.__class__))
            setattr(m, 'get_mask', get_mask_by_score.__get__(m, m.__class__))

            if isinstance(m, nn.Linear):
                m.dim_in = (0)
                m.dim_out = (1)
                m.view_in = (1, -1)
                m.view_out = (-1, 1)
                setattr(m, 'base_func', linear_forward.__get__(m, m.__class__))
            else:
                m.dim_in = (0,2,3)
                m.dim_out = (1,2,3)
                m.view_in = (1, -1, 1, 1)
                m.view_out = (-1, 1, 1, 1)
                setattr(m, 'base_func', m._conv_forward)

            setattr(m, 'forward', NPB_forward.__get__(m, m.__class__))

        elif isinstance(m, Residual):
            setattr(m, 'original_forward', m.forward)
            setattr(m, 'forward', NPB_residual_forward.__get__(m, m.__class__))
        elif isinstance(m, nn.MaxPool2d) or isinstance(m, nn.AvgPool2d) or isinstance(m, nn.Flatten):
            setattr(m, 'original_forward', m.forward)
            setattr(m, 'forward', NPB_stable_forward.__get__(m, m.__class__))
        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.LogSoftmax) or isinstance(m, nn.ReLU) or isinstance(m, nn.Dropout):
            setattr(m, 'original_forward', m.forward)
            setattr(m, 'forward', NPB_dummy_forward.__get__(m, m.__class__))

    model.NPB_modules = NPB_modules
    model.NPB_params = [m.weight for m in NPB_modules]
    ERK_sparsify(model, 1-args.density)

def ERK_sparsify(model, sparsity=0.9):
    # print('initialize by ERK')
    density = 1 - sparsity
    erk_power_scale = 1

    total_params = 0
    for m in model.NPB_modules:
        total_params += m.score.numel()
    is_epsilon_valid = False

    dense_layers = set()
    while not is_epsilon_valid:
        divisor = 0
        rhs = 0
        for m in model.NPB_modules:
            m.raw_probability = 0
            n_param = np.prod(m.score.shape)
            n_zeros = n_param * (1 - density)
            n_ones = n_param * density

            if m in dense_layers:
                rhs -= n_zeros
            else:
                rhs += n_ones
                m.raw_probability = (np.sum(m.score.shape) / np.prod(m.score.shape)) ** erk_power_scale
                divisor += m.raw_probability * n_param

        epsilon = rhs / divisor
        max_prob = np.max([m.raw_probability for m in model.NPB_modules])
        # print([m.raw_probability for m in model.NPB_modules])
        max_prob_one = max_prob * epsilon
        if max_prob_one > 1:
            is_epsilon_valid = False
            for m in model.NPB_modules:
                if m.raw_probability == max_prob:
                    dense_layers.add(m)
        else:
            is_epsilon_valid = True

    total_nonzero = 0.0
    # With the valid epsilon, we can set sparsities of the remaning layers.
    for i, m in enumerate(model.NPB_modules):
        n_param = np.prod(m.score.shape)
        if m in dense_layers:
            m.sparsity = 0
        else:
            probability_one = epsilon * m.raw_probability
            m.sparsity = 1 - probability_one
        m.num_zeros = int((m.sparsity) * m.score.numel())
        # print(
        #     f"layer: {i}, shape: {m.score.shape}, sparsity: {m.sparsity}"
        # )
        total_nonzero += (1-m.sparsity) * m.score.numel()
    print(f"Overall sparsity {1-total_nonzero / total_params}")
