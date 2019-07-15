"""Modified from https://github.com/JiahuiYu/slimmable_networks/blob/master/utils/model_profiling.py"""
import logging
import functools
import numpy as np
import time
import torch
import torch.nn as nn
import models.mobilenet_supernet as ms
import models.mobilenet_base as mb

model_profiling_hooks = []
model_profiling_speed_hooks = []

name_space = 95
params_space = 15
macs_space = 15
seconds_space = 15


class Timer(object):

    def __init__(self, verbose=False):
        self.verbose = verbose
        self.start = None
        self.end = None

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.time = self.end - self.start
        if self.verbose:
            print('Elapsed time: %f ms.' % self.time)


def get_params(self):
    """get number of params in module"""
    return np.sum([np.prod(list(w.size())) for w in self.parameters()])


def run_forward(self, input, num_forwards=10):
    if num_forwards <= 0:
        return 0.0
    with Timer() as t:
        for _ in range(num_forwards):
            self.forward(*input)
            torch.cuda.synchronize()
    return int(t.time * 1e9 / num_forwards)


def conv_module_name_filter(name):
    """filter module name to have a short view"""
    filters = {
        'kernel_size': 'k',
        'stride': 's',
        'padding': 'pad',
        'bias': 'b',
        'groups': 'g',
    }
    for k in filters:
        name = name.replace(k, filters[k])
    return name


def module_profiling(self, input, output, num_forwards, verbose):

    def add_sub(m, sub_op):
        m.n_macs += getattr(sub_op, 'n_macs', 0)
        m.n_params += getattr(sub_op, 'n_params', 0)
        m.n_seconds += getattr(sub_op, 'n_seconds', 0)

    _run_forward = functools.partial(run_forward, num_forwards=num_forwards)
    ins = input[0].size()
    outs = output.size()
    # NOTE: There are some difference between type and isinstance, thus please
    # be careful.
    t = type(self)
    self._profiling_input_size = ins
    self._profiling_output_size = outs
    if isinstance(self, nn.Conv2d):
        self.n_macs = (ins[1] * outs[1] * self.kernel_size[0] *
                       self.kernel_size[1] * outs[2] * outs[3] //
                       self.groups) * outs[0]
        self.n_params = get_params(self)
        self.n_seconds = _run_forward(self, input)
        self.name = conv_module_name_filter(self.__repr__())
    elif isinstance(self, nn.ConvTranspose2d):
        self.n_macs = (ins[1] * outs[1] * self.kernel_size[0] *
                       self.kernel_size[1] * outs[2] * outs[3] //
                       self.groups) * outs[0]
        self.n_params = get_params(self)
        self.n_seconds = _run_forward(self, input)
        self.name = conv_module_name_filter(self.__repr__())
    elif isinstance(self, nn.Linear):
        self.n_macs = ins[1] * outs[1] * outs[0]
        self.n_params = get_params(self)
        self.n_seconds = _run_forward(self, input)
        self.name = self.__repr__()
    elif isinstance(self, nn.AvgPool2d):
        # NOTE: this function is correct only when stride == kernel size
        self.n_macs = ins[1] * ins[2] * ins[3] * ins[0]
        self.n_params = 0
        self.n_seconds = _run_forward(self, input)
        self.name = self.__repr__()
    elif isinstance(self, nn.AdaptiveAvgPool2d):
        # NOTE: this function is correct only when stride == kernel size
        self.n_macs = ins[1] * ins[2] * ins[3] * ins[0]
        self.n_params = 0
        self.n_seconds = _run_forward(self, input)
        self.name = self.__repr__()
    elif isinstance(self, mb.SqueezeAndExcitation):
        # TODO(meijieru): fix time cal
        self.n_macs = ins[1] * ins[2] * ins[3] * ins[0]
        self.n_params = 0
        self.n_seconds = 0
        add_sub(self, self.se_reduce)
        add_sub(self, self.se_expand)
        self.name = self.__repr__()
    elif isinstance(self, mb.InvertedResidualChannels):
        self.n_macs = 0
        self.n_params = 0
        self.n_seconds = 0
        for op in self.ops:
            add_sub(self, op)
        self.name = self.__repr__()
    elif isinstance(self, mb.InvertedResidualChannelsFused):
        self.n_macs = 0
        self.n_params = 0
        self.n_seconds = 0
        for op in self.depth_ops:
            add_sub(self, op)
        add_sub(self, self.expand_conv)
        add_sub(self, self.project_conv)
        add_sub(self, self.se_op)
        self.name = self.__repr__()
    else:
        # This works only in depth-first travel of modules.
        self.n_macs = 0
        self.n_params = 0
        self.n_seconds = 0
        num_children = 0
        for m in self.children():
            self.n_macs += getattr(m, 'n_macs', 0)
            self.n_params += getattr(m, 'n_params', 0)
            self.n_seconds += getattr(m, 'n_seconds', 0)
            num_children += 1
        ignore_zeros_t = [
            nn.BatchNorm2d,
            nn.Dropout2d,
            nn.Dropout,
            nn.Sequential,
            nn.ReLU6,
            nn.ReLU,
            mb.Swish,
            mb.Narrow,
            mb.Identity,
            nn.MaxPool2d,
            nn.modules.padding.ZeroPad2d,
            nn.modules.activation.Sigmoid,
        ]
        if (not getattr(self, 'ignore_model_profiling', False) and
                self.n_macs == 0 and t not in ignore_zeros_t):
            logging.info('WARNING: leaf module {} has zero n_macs.'.format(
                type(self)))
        return
    if verbose:
        logging.info(
            self.name.ljust(name_space, ' ') +
            '{:,}'.format(self.n_params).rjust(params_space, ' ') +
            '{:,}'.format(self.n_macs).rjust(macs_space, ' ') +
            '{:,}'.format(self.n_seconds).rjust(seconds_space, ' '))
    return


def add_profiling_hooks(m, num_forwards, verbose):
    global model_profiling_hooks
    model_profiling_hooks.append(
        m.register_forward_hook(lambda m, input, output: module_profiling(
            m, input, output, num_forwards, verbose=verbose)))


def remove_profiling_hooks():
    global model_profiling_hooks
    for h in model_profiling_hooks:
        h.remove()
    model_profiling_hooks = []


def model_profiling(model,
                    height,
                    width,
                    batch=1,
                    channel=3,
                    use_cuda=True,
                    num_forwards=10,
                    verbose=True):
    """ Pytorch model profiling with input image size
    (batch, channel, height, width).
    The function exams the number of multiply-accumulates (n_macs).

    Args:
        model: pytorch model
        height: int
        width: int
        batch: int
        channel: int
        use_cuda: bool

    Returns:
        macs: int
        params: int

    """
    model.eval()
    data = torch.rand(batch, channel, height, width)
    origin_device = next(model.parameters()).device
    device = torch.device("cuda" if use_cuda else "cpu")
    model = model.to(device)
    data = data.to(device)
    model.apply(lambda m: add_profiling_hooks(m, num_forwards, verbose=verbose))
    if verbose:
        logging.info('Item'.ljust(name_space, ' ') +
                     'params'.rjust(macs_space, ' ') +
                     'macs'.rjust(macs_space, ' ') +
                     'nanosecs'.rjust(seconds_space, ' '))
        logging.info(''.center(
            name_space + params_space + macs_space + seconds_space, '-'))
    with torch.no_grad():
        model(data)
    if verbose:
        logging.info(''.center(
            name_space + params_space + macs_space + seconds_space, '-'))
        logging.info('Total'.ljust(name_space, ' ') +
                     '{:,}'.format(model.n_params).rjust(params_space, ' ') +
                     '{:,}'.format(model.n_macs).rjust(macs_space, ' ') +
                     '{:,}'.format(model.n_seconds).rjust(seconds_space, ' '))
    remove_profiling_hooks()
    model = model.to(origin_device)
    return model.n_macs, model.n_params
