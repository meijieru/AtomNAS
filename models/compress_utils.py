"""TODO(meijieru): further comment"""
import warnings
import itertools
import functools
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.common import add_prefix


def _scatter_by_bool(l, mask, pad=None):
    idx = 0
    res = []
    for val in mask:
        if val:
            res.append(l[idx])
            idx += 1
        else:
            res.append(pad)
    assert len(l) == idx
    return res


def _find_only_one(func, iterable):
    res = list(filter(func, iterable))
    assert len(res) == 1
    return res[0]


def _mask_along_dim(lhs, rhs, mask, dim=0):
    if dim == 0:
        lhs.data.copy_(rhs.data[mask])
    elif dim == 1:
        lhs.data.copy_(rhs.data[:, mask])
    else:
        raise NotImplementedError()


def _copy(lhs, rhs, *args):
    lhs.data.copy_(rhs.data)


def build_default_info(m_new,
                       m_old,
                       mask,
                       attr,
                       mask_hook,
                       var_type='variable',
                       prefix_new=None,
                       prefix_old=None):

    def _build_name(prefix):
        return add_prefix(attr, prefix)

    assert var_type in ['variable', 'buffer']
    info = {
        'var_old_name': _build_name(prefix_old),
        'var_old': getattr(m_old, attr),
        'type': var_type,
        'mask': mask,
        'module_class': type(m_old)
    }
    if m_new is not None:
        info.update({
            'var_new_name': _build_name(prefix_new),
            'var_new': getattr(m_new, attr),
            'mask_hook': mask_hook,
        })
    return info


def compress_conv(m_new, m_old, mask, dim, prefix_new=None, prefix_old=None):
    assert m_new is None or isinstance(m_new, nn.Conv2d)
    assert isinstance(m_old, nn.Conv2d)
    assert dim in [0, 1]

    build_info = functools.partial(build_default_info,
                                   m_new,
                                   m_old,
                                   mask,
                                   prefix_new=prefix_new,
                                   prefix_old=prefix_old)
    infos = [build_info('weight', functools.partial(_mask_along_dim, dim=dim))]
    if m_old.bias is not None:
        if dim == 0:  # reduce output channels
            infos.append(build_info('bias', _mask_along_dim))
        else:
            infos.append(build_info('bias', _copy))
    return infos


def compress_bn(m_new, m_old, mask, prefix_new=None, prefix_old=None):
    assert m_new is None or isinstance(m_new, nn.BatchNorm2d)
    assert isinstance(m_old, nn.BatchNorm2d)
    assert m_new is None or m_new.affine == m_old.affine

    infos = []
    if m_old.affine:
        build_info = functools.partial(build_default_info,
                                       m_new,
                                       m_old,
                                       mask,
                                       prefix_new=prefix_new,
                                       prefix_old=prefix_old)
        infos.append(build_info('weight', _mask_along_dim))
        infos.append(build_info('bias', _mask_along_dim))
    if m_old.track_running_stats:
        build_buffer_info = functools.partial(build_default_info,
                                              m_new,
                                              m_old,
                                              mask,
                                              var_type='buffer',
                                              prefix_new=prefix_new,
                                              prefix_old=prefix_old)
        infos.append(build_buffer_info('running_var', _mask_along_dim))
        infos.append(build_buffer_info('running_mean', _mask_along_dim))
        infos.append(build_buffer_info('num_batches_tracked', _copy))
    return infos


def _adjust_bn_mean(info):
    adjust_infos = info['post_hook_params']
    running_mean = info['var_new']
    for adjust_info in adjust_infos:
        active_fn = adjust_info['active_fn']()
        bias = adjust_info['bias']['var_old']
        following_conv_weight = adjust_info['following_conv_weight']['var_old']
        mask = adjust_info['bias']['mask']
        assert id(mask) == id(adjust_info['following_conv_weight']['mask'])
        _, _, h, w = following_conv_weight.size()
        assert h == 1 and w == 1, 'Not equal due to padding!'
        with torch.no_grad():
            tmp = F.conv2d(active_fn(bias[~mask].view(1, -1, 1, 1)),
                           following_conv_weight[:, ~mask])
            print('adjust', tmp)
            running_mean.data.sub_(torch.squeeze(tmp))


def adjust_bn(m_new, m_old, post_hook_params, **kwargs):
    mask = torch.ones_like(m_new.weight, dtype=torch.bool)
    infos = compress_bn(m_new, m_old, mask, **kwargs)
    info_mean = _find_only_one(
        lambda info: 'running_mean' in info['var_old_name'], infos)
    info_mean['post_hook_params'] = post_hook_params
    info_mean['post_hook'] = _adjust_bn_mean
    return infos


def compress_conv_bn_relu(m_new,
                          m_old,
                          mask,
                          prefix_new=None,
                          prefix_old=None,
                          dim=0):
    import models.mobilenet_base as mb

    assert m_new is None or isinstance(m_new, mb.ConvBNReLU)
    assert isinstance(m_old, mb.ConvBNReLU)

    old_children = list(m_old.children())
    if m_new is None:
        new_children = [None for _ in old_children]
    else:
        new_children = list(m_new.children())
    conv_infos = compress_conv(new_children[0],
                               old_children[0],
                               mask,
                               dim=dim,
                               prefix_new='{}.0'.format(prefix_new),
                               prefix_old='{}.0'.format(prefix_old))
    bn_infos = compress_bn(new_children[1],
                           old_children[1],
                           mask,
                           prefix_new='{}.1'.format(prefix_new),
                           prefix_old='{}.1'.format(prefix_old))
    return conv_infos + bn_infos


def copmress_inverted_residual_channels(m,
                                        masks,
                                        ema=None,
                                        optimizer=None,
                                        prune_info=None,
                                        prefix=None,
                                        verbose=False):

    def update(infos):
        for info in infos:
            if optimizer is not None and info['type'] != 'buffer':
                optimizer.compress_mask(info, verbose=verbose)
            if ema is not None and 'num_batches_tracked' not in info[
                    'var_old_name']:
                ema.compress_mask(info, verbose=verbose)
            if prune_info is not None and issubclass(
                    info['module_class'],
                    nn.BatchNorm2d) and info['type'] == 'variable':
                if prune_info.compress_check_exist(info):
                    prune_info.compress_mask(info, verbose=verbose)
            info['mask_hook'](info['var_new'], info['var_old'], info['mask'])
            if 'post_hook' in info:
                # FIXME(meijieru): bn adjust
                warnings.warn('Do not adjust bn mean!!!')
                # info['post_hook'](info)

    def clean(infos):
        for info in infos:
            if optimizer is not None and info['type'] != 'buffer':
                optimizer.compress_drop(info, verbose=verbose)
            if ema is not None and 'num_batches_tracked' not in info[
                    'var_old_name']:
                ema.compress_drop(info, verbose=verbose)
            if prune_info is not None and issubclass(
                    info['module_class'],
                    nn.BatchNorm2d) and info['type'] == 'variable':
                if prune_info.compress_check_exist(info):
                    prune_info.compress_drop(info, verbose=verbose)

    assert len(m.kernel_sizes) == len(masks)
    hidden_dims = [mask.detach().sum().item() for mask in masks]
    indices = torch.arange(len(m.ops))
    keeps = [num_remain > 0 for num_remain in hidden_dims]
    m.channels, m.kernel_sizes = [
        list(itertools.compress(x, keeps))
        for x in [hidden_dims, m.kernel_sizes]
    ]
    new_ops, new_pw_bn = m._build(m.channels, m.kernel_sizes, m.expand)
    new_indices = torch.arange(len(new_ops))
    if m.expand:
        idx_depth = 1
        idx_proj = 2
    else:
        idx_depth = 0
        idx_proj = 1

    new_ops_padded = _scatter_by_bool(new_ops, keeps)
    new_indices_padded = _scatter_by_bool(new_indices, keeps)

    # update ema, optimizer, module
    pending_clean_infos = []
    pending_adjust_infos = []
    adjust_infos = []
    for new_op, new_indice, old_op, old_indice, mask in zip(
            new_ops_padded, new_indices_padded, m.ops, indices, masks):
        old_op_children = list(old_op.children())
        if new_op is None:  # drop old
            new_op_children = [None for _ in old_op_children]
            pending_infos = pending_clean_infos
        else:
            new_op_children = list(new_op.children())
            pending_infos = pending_adjust_infos
        if m.expand:
            expand_infos = compress_conv_bn_relu(
                new_op_children[0], old_op_children[0], mask,
                add_prefix('ops.{}.0'.format(new_indice), prefix),
                add_prefix('ops.{}.0'.format(old_indice), prefix))
            pending_infos.append(expand_infos)
        depth_infos = compress_conv_bn_relu(
            new_op_children[idx_depth], old_op_children[idx_depth], mask,
            add_prefix('ops.{}.{}'.format(new_indice, idx_depth), prefix),
            add_prefix('ops.{}.{}'.format(old_indice, idx_depth), prefix))
        pending_infos.append(depth_infos)
        proj_infos = compress_conv(
            new_op_children[idx_proj],
            old_op_children[idx_proj],
            mask,
            dim=1,
            prefix_new=add_prefix('ops.{}.{}'.format(new_indice, idx_proj),
                                  prefix),
            prefix_old=add_prefix('ops.{}.{}'.format(old_indice, idx_proj),
                                  prefix))
        pending_infos.append(proj_infos)

        adjust_info = {'active_fn': m.active_fn}
        adjust_info['bias'] = _find_only_one(
            lambda info: issubclass(info['module_class'], nn.BatchNorm2d) and
            'bias' in info['var_old_name'], depth_infos)
        adjust_info['following_conv_weight'] = _find_only_one(
            lambda info: issubclass(info['module_class'], nn.Conv2d) and
            'weight' in info['var_old_name'], proj_infos)
        adjust_infos.append(adjust_info)
    prefix_pw_bn = add_prefix('pw_bn', prefix)
    pw_bn_infos = adjust_bn(new_pw_bn,
                            m.pw_bn,
                            adjust_infos,
                            prefix_new=prefix_pw_bn,
                            prefix_old=prefix_pw_bn)

    if ema is not None:
        ema.compress_start()
    if prune_info is not None:
        prune_info.compress_start()
    update(pw_bn_infos)  # NOTE: must do before following for ema
    for infos in pending_adjust_infos:
        update(infos)
    for infos in pending_clean_infos:  # remove non-use last
        clean(infos)

    del m.ops
    del m.pw_bn
    m.ops, m.pw_bn = new_ops, new_pw_bn
