"""Common utilities for mobilenet."""
import abc
import collections
import logging
import functools
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import models.compress_utils as cu
from utils.common import add_prefix
from utils.common import get_device


def _make_divisible(v, divisor, min_value=None):
    """Make channels divisible to divisor.

    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class CheckpointModule(nn.Module, metaclass=abc.ABCMeta):
    """Discard mid-result using checkpoint."""

    def __init__(self, use_checkpoint=True):
        super(CheckpointModule, self).__init__()
        self._use_checkpoint = use_checkpoint

    def forward(self, *args, **kwargs):
        from torch.utils.checkpoint import checkpoint
        if self._use_checkpoint:
            return checkpoint(self._forward, *args, **kwargs)
        return self._forward(*args, **kwargs)

    @abc.abstractmethod
    def _forward(self, *args, **kwargs):
        pass


class Identity(nn.Module):
    """Module proxy for null op."""

    def forward(self, x):
        return x


class Narrow(nn.Module):
    """Module proxy for `torch.narrow`."""

    def __init__(self, dimension, start, length):
        super(Narrow, self).__init__()
        self.dimension = dimension
        self.start = start
        self.length = length

    def forward(self, x):
        return x.narrow(self.dimension, self.start, self.length)


class Swish(nn.Module):
    """Swish activation function.

    See: https://arxiv.org/abs/1710.05941
    NOTE: Will consume much more GPU memory compared with inplaced ReLU.
    """

    def forward(self, x):
        return x * torch.sigmoid(x)


class HSwish(object):
    """Hard Swish activation function.

    See: https://arxiv.org/abs/1905.02244
    """

    def forward(self, x):
        return x * F.relu6(x + 3, True).div_(6)


class SqueezeAndExcitation(nn.Module):
    """Squeeze-and-Excitation module.

    See: https://arxiv.org/abs/1709.01507
    """

    def __init__(self, n_feature, n_hidden, spatial_dims=[2, 3],
                 active_fn=None):
        super(SqueezeAndExcitation, self).__init__()
        self.n_feature = n_feature
        self.n_hidden = n_hidden
        self.spatial_dims = spatial_dims
        self.se_reduce = nn.Conv2d(n_feature, n_hidden, 1, bias=True)
        self.se_expand = nn.Conv2d(n_hidden, n_feature, 1, bias=True)
        self.active_fn = active_fn()

    def forward(self, x):
        se_tensor = x.mean(self.spatial_dims, keepdim=True)
        se_tensor = self.se_expand(self.active_fn(self.se_reduce(se_tensor)))
        return torch.sigmoid(se_tensor) * x

    def __repr__(self):
        return '{}({}, {}, spatial_dims={}, active_fn={})'.format(
            self._get_name(), self.n_feature, self.n_hidden, self.spatial_dims,
            self.active_fn)


class ConvBNReLU(nn.Sequential):
    """Convolution-BatchNormalization-ActivateFn."""

    def __init__(self,
                 in_planes,
                 out_planes,
                 kernel_size=3,
                 stride=1,
                 groups=1,
                 active_fn=None,
                 batch_norm_kwargs=None):
        if batch_norm_kwargs is None:
            batch_norm_kwargs = {}
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes,
                      out_planes,
                      kernel_size,
                      stride,
                      padding,
                      groups=groups,
                      bias=False),
            nn.BatchNorm2d(out_planes, **batch_norm_kwargs), active_fn())


class InvertedResidualChannelsFused(nn.Module):
    """Speedup version of `InvertedResidualChannels` by fusing small kernels.

    NOTE: It may consume more GPU memory.
    Support `Squeeze-and-Excitation`.
    """

    def __init__(self,
                 inp,
                 oup,
                 stride,
                 channels,
                 kernel_sizes,
                 expand,
                 active_fn=None,
                 batch_norm_kwargs=None,
                 se_ratio=None):
        super(InvertedResidualChannelsFused, self).__init__()
        assert stride in [1, 2]
        assert len(channels) == len(kernel_sizes)

        self.input_dim = inp
        self.output_dim = oup
        self.expand = expand
        self.stride = stride
        self.kernel_sizes = kernel_sizes
        self.channels = channels
        self.use_res_connect = self.stride == 1 and inp == oup
        self.batch_norm_kwargs = batch_norm_kwargs
        self.active_fn = active_fn
        self.se_ratio = se_ratio

        (self.expand_conv, self.depth_ops, self.project_conv,
         self.se_op) = self._build(channels, kernel_sizes, expand, se_ratio)

    def _build(self, hidden_dims, kernel_sizes, expand, se_ratio):
        _batch_norm_kwargs = self.batch_norm_kwargs \
            if self.batch_norm_kwargs is not None else {}

        hidden_dim_total = sum(hidden_dims)
        if self.expand:
            # pw
            expand_conv = ConvBNReLU(self.input_dim,
                                     hidden_dim_total,
                                     kernel_size=1,
                                     batch_norm_kwargs=_batch_norm_kwargs,
                                     active_fn=self.active_fn)
        else:
            expand_conv = Identity()

        narrow_start = 0
        depth_ops = nn.ModuleList()
        for k, hidden_dim in zip(kernel_sizes, hidden_dims):
            layers = []
            if expand:
                layers.append(Narrow(1, narrow_start, hidden_dim))
                narrow_start += hidden_dim
            else:
                if hidden_dim != self.input_dim:
                    raise RuntimeError('uncomment this for search_first model')
                logging.warning(
                    'uncomment this for previous trained search_first model')
            layers.extend([
                # dw
                ConvBNReLU(hidden_dim,
                           hidden_dim,
                           kernel_size=k,
                           stride=self.stride,
                           groups=hidden_dim,
                           batch_norm_kwargs=_batch_norm_kwargs,
                           active_fn=self.active_fn),
            ])
            depth_ops.append(nn.Sequential(*layers))
        project_conv = nn.Sequential(
            nn.Conv2d(hidden_dim_total, self.output_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.output_dim, **_batch_norm_kwargs))

        if expand and narrow_start != hidden_dim_total:
            raise ValueError('Part of expanded are not used')

        if se_ratio is not None:
            se_op = SqueezeAndExcitation(hidden_dim_total,
                                         int(round(self.input_dim * se_ratio)),
                                         active_fn=self.active_fn)
        else:
            se_op = Identity()
        return expand_conv, depth_ops, project_conv, se_op

    def get_depthwise_bn(self):
        """Get `[module]` list of BN after depthwise convolution."""
        return list(self.get_named_depthwise_bn().values())

    def get_named_depthwise_bn(self, prefix=None):
        """Get `{name: module}` pairs of BN after depthwise convolution."""
        res = collections.OrderedDict()
        for i, op in enumerate(self.depth_ops):
            children = list(op.children())
            if self.expand:
                idx_op = 1
            else:
                raise RuntimeError('Not search_first')
            conv_bn_relu = children[idx_op]
            assert isinstance(conv_bn_relu, ConvBNReLU)
            conv_bn_relu = list(conv_bn_relu.children())
            _, bn, _ = conv_bn_relu
            assert isinstance(bn, nn.BatchNorm2d)
            name = 'depth_ops.{}.{}.1'.format(i, idx_op)
            name = add_prefix(name, prefix)
            res[name] = bn
        return res

    def forward(self, x):
        res = self.expand_conv(x)
        res = [op(res) for op in self.depth_ops]
        if len(res) != 1:
            res = torch.cat(res, dim=1)
        else:
            res = res[0]
        res = self.se_op(res)
        res = self.project_conv(res)
        if self.use_res_connect:
            return x + res
        return res

    def __repr__(self):
        return ('{}({}, {}, channels={}, kernel_sizes={}, expand={}, stride={},'
                ' se_ratio={})').format(self._get_name(), self.input_dim,
                                        self.output_dim, self.channels,
                                        self.kernel_sizes, self.expand,
                                        self.stride, self.se_ratio)


class InvertedResidualChannels(nn.Module):
    """MobiletNetV2 building block."""

    def __init__(self,
                 inp,
                 oup,
                 stride,
                 channels,
                 kernel_sizes,
                 expand,
                 active_fn=None,
                 batch_norm_kwargs=None):
        super(InvertedResidualChannels, self).__init__()
        assert stride in [1, 2]
        assert len(channels) == len(kernel_sizes)

        self.input_dim = inp
        self.output_dim = oup
        self.expand = expand
        self.stride = stride
        self.kernel_sizes = kernel_sizes
        self.channels = channels
        self.use_res_connect = self.stride == 1 and inp == oup
        self.batch_norm_kwargs = batch_norm_kwargs
        self.active_fn = active_fn

        self.ops, self.pw_bn = self._build(channels, kernel_sizes, expand)

    def _build(self, hidden_dims, kernel_sizes, expand):
        _batch_norm_kwargs = self.batch_norm_kwargs \
            if self.batch_norm_kwargs is not None else {}

        narrow_start = 0
        ops = nn.ModuleList()
        for k, hidden_dim in zip(kernel_sizes, hidden_dims):
            layers = []
            if expand:
                # pw
                layers.append(
                    ConvBNReLU(self.input_dim,
                               hidden_dim,
                               kernel_size=1,
                               batch_norm_kwargs=_batch_norm_kwargs,
                               active_fn=self.active_fn))
            else:
                if hidden_dim != self.input_dim:
                    raise RuntimeError('uncomment this for search_first model')
                logging.warning(
                    'uncomment this for previous trained search_first model')
                # layers.append(Narrow(1, narrow_start, hidden_dim))
                narrow_start += hidden_dim
            layers.extend([
                # dw
                ConvBNReLU(hidden_dim,
                           hidden_dim,
                           kernel_size=k,
                           stride=self.stride,
                           groups=hidden_dim,
                           batch_norm_kwargs=_batch_norm_kwargs,
                           active_fn=self.active_fn),
                # pw-linear
                nn.Conv2d(hidden_dim, self.output_dim, 1, 1, 0, bias=False),
                # nn.BatchNorm2d(oup, **batch_norm_kwargs),
            ])
            ops.append(nn.Sequential(*layers))
        pw_bn = nn.BatchNorm2d(self.output_dim, **_batch_norm_kwargs)

        if not expand and narrow_start != self.input_dim:
            raise ValueError('Part of input are not used')
        return ops, pw_bn

    def get_depthwise_bn(self):
        """Get `[module]` list of BN after depthwise convolution."""
        return list(self.get_named_depthwise_bn().values())

    def get_named_depthwise_bn(self, prefix=None):
        """Get `{name: module}` pairs of BN after depthwise convolution."""
        res = collections.OrderedDict()
        for i, op in enumerate(self.ops):
            children = list(op.children())
            if self.expand:
                idx_op = 1
            else:
                idx_op = 0
            conv_bn_relu = children[idx_op]
            assert isinstance(conv_bn_relu, ConvBNReLU)
            conv_bn_relu = list(conv_bn_relu.children())
            _, bn, _ = conv_bn_relu
            assert isinstance(bn, nn.BatchNorm2d)
            name = 'ops.{}.{}.1'.format(i, idx_op)
            name = add_prefix(name, prefix)
            res[name] = bn
        return res

    def forward(self, x):
        if len(self.ops) == 0:
            if not self.use_res_connect:
                logging.warning(
                    'The whole block is pruned without skip connection')
            return x

        tmp = sum([op(x) for op in self.ops])
        tmp = self.pw_bn(tmp)
        if self.use_res_connect:
            return x + tmp
        return tmp

    def __repr__(self):
        return ('{}({}, {}, channels={}, kernel_sizes={}, expand={},'
                ' stride={})').format(self._get_name(), self.input_dim,
                                      self.output_dim, self.channels,
                                      self.kernel_sizes, self.expand,
                                      self.stride)

    def compress_by_mask(self, masks, **kwargs):
        """Regenerate internal compute graph given alive masks."""
        device = get_device(self.pw_bn)
        cu.copmress_inverted_residual_channels(self, masks, **kwargs)
        self.to(device)

    def compress_by_threshold(self, threshold, **kwargs):
        """Regenerate internal compute graph by discarding dead atomic blocks.
        """
        masks = [
            bn.weight.detach().abs() > threshold
            for bn in self.get_depthwise_bn()
        ]
        self.compress_by_mask(masks, **kwargs)


def get_active_fn(name):
    """Select activation function."""
    active_fn = {
        'nn.ReLU6': functools.partial(nn.ReLU6, inplace=True),
        'nn.ReLU': functools.partial(nn.ReLU, inplace=True),
        'nn.Swish': Swish,
        'nn.HSwish': HSwish,
    }[name]
    return active_fn


def get_block(name):
    """Select building block."""
    return {
        'InvertedResidualChannels': InvertedResidualChannels,
        'InvertedResidualChannelsFused': InvertedResidualChannelsFused
    }[name]


def init_weights_slimmable(m):
    """Slimmable network style initialization."""
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, 0, 0.01)
        nn.init.zeros_(m.bias)


def init_weights_mnas(m):
    """MnasNet style initialization."""
    if isinstance(m, nn.Conv2d):
        if m.groups == m.in_channels:  # depthwise conv
            fan_out = m.weight[0][0].numel()
        else:
            _, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight)
        gain = nn.init.calculate_gain('relu')
        std = gain / math.sqrt(fan_out)
        nn.init.normal_(m.weight, 0.0, std)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        _, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight)
        init_range = 1.0 / np.sqrt(fan_out)
        nn.init.uniform_(m.weight, -init_range, init_range)
        nn.init.zeros_(m.bias)


def output_network(model):
    """Output network kwargs in `searched_network` style."""
    model_kwargs = {
        key: getattr(model, key) for key in [
            'input_channel', 'last_channel', 'width_mult', 'round_nearest',
            'active_fn', 'num_classes'
        ]
    }
    blocks = list(model.get_named_block_list().values())

    res = []
    for block in blocks:
        res.append([
            block.output_dim, 1, block.stride, block.kernel_sizes,
            block.channels, block.expand
        ])
    model_kwargs['inverted_residual_setting'] = res
    return model_kwargs


def _get_named_block_list(m):
    """Get `{name: module}` dictionary for inverted residual blocks."""
    blocks = list(m.features.named_children())
    blocks = blocks[1:-2]
    return collections.OrderedDict([
        ('features.{}'.format(name), block) for name, block in blocks
    ])
