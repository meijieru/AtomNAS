"""Implement resource-aware atomic block selection."""
import collections
import numbers
import itertools
import logging
import torch
import torch.nn as nn
import models.mobilenet_base as mb


class PruneInfo(object):
    """Information for resource-aware atomic block selection."""

    def __init__(self, names, penalties):
        """Init property for weights to be selected."""
        assert len(names) == len(penalties)
        self._info = collections.OrderedDict((k, {
            'compress_masked': False,
            'penalty': v
        }) for k, v in zip(names, penalties))

    def add_info_list(self, name, values):
        """Add related property."""
        assert len(values) == len(self.weight)
        for k, v in zip(self.weight, values):
            self._info[k][name] = v

    def get_info_list(self, name):
        """Get property by name."""
        return [v[name] for v in self._info.values()]

    @property
    def weight(self):
        return list(self._info.keys())

    @property
    def penalty(self):
        return self.get_info_list('penalty')

    def compress_start(self):
        """Setup info used for dynamic network shrinkage.

        Typical dynamic network shrinkage for `PruneInfo`:
            ```
            prune_info.compress_start()
            for info in infos_mask:
                if prune_info.compress_check_exist(info):
                    prune_info.compress_mask(*args, **kwargs)
            for info in infos_drop:
                if prune_info.compress_check_exist(info):
                    prune_info.compress_drop(*args, **kwargs)
            ```
        """
        for val in self._info.values():
            val['compress_masked'] = False

    def compress_check_exist(self, info):
        """Check whether old var is a candidate."""
        name = info['var_old_name']
        return name in self._info

    def compress_mask(self, info, verbose=False):
        """Adjust parameters values by masks for dynamic network shrinkage."""
        var_old_name = info['var_old_name']
        var_new_name = info['var_new_name']
        if verbose:
            logging.info('PruneInfo compress: {} -> {}'.format(
                var_old_name, var_new_name))
        if self._info[var_old_name]['compress_masked']:
            raise RuntimeError('May have dependencies in compress')
        if var_new_name in self._info and self._info[var_new_name][
                'compress_masked']:
            raise RuntimeError('Compress {} twice'.format(var_new_name))
        self._info[var_new_name] = self._info.pop(var_old_name)
        self._info[var_new_name]['compress_masked'] = True

    def compress_drop(self, info, verbose=False):
        """Remove unused parameters for dynamic network shrinkage."""
        name = info['var_old_name']
        if verbose:
            logging.info('PruneInfo drop: {}'.format(name))
        if self._info[name]['compress_masked']:
            if verbose:
                logging.info('PruneInfo drop: {} skipped'.format(name))
        else:
            return self._info.pop(name)


def get_bn_to_prune(model, flags, verbose=True):
    """Init information for atomic block selection.

    Args:
        model: A model with method `get_named_block_list` which return all
            MobileNet V2 blocks with their names in `state_dict`.
        flags: Configuration class.
        verbose: Log verbose info.

    Returns:
        An instance of `PruneInfo`.
    """
    bn_prune_filter = flags.get('bn_prune_filter', None)
    if bn_prune_filter in ['expansion_only', 'expansion_only_skip_expand1']:
        # resource aware channel selection
        weights = []
        penalties = []
        for name, m in model.get_named_block_list().items():
            if isinstance(m, mb.InvertedResidualChannels):
                # only the first block could be non expand
                if bn_prune_filter == 'expansion_only_skip_expand1' and not m.expand:
                    continue

                for op, (bn_name, bn) in zip(
                        m.ops,
                        m.get_named_depthwise_bn(prefix=name).items()):
                    hidden_channel = bn.weight.numel()
                    penalties.append(
                        (hidden_channel, op.n_macs / hidden_channel))
                    weights.append('{}.weight'.format(bn_name))
        per_channel_flops = [val[1] for val in penalties]
        numel_total = sum(val[0] for val in penalties)
        penalty_normalizer = sum([numel * val for numel, val in penalties
                                 ]) / (numel_total + 1e-5)
        penalties = [val / penalty_normalizer for (_, val) in penalties]
    elif bn_prune_filter in ['equal_penalty_skip_expand1']:
        # baseline for table 2, network slimming like
        weights = []
        penalties = []
        per_channel_flops = []
        for name, m in model.get_named_block_list().items():
            if isinstance(m, mb.InvertedResidualChannels):
                if bn_prune_filter == 'equal_penalty_skip_expand1' and not m.expand:
                    continue

                for op, (bn_name, bn) in zip(
                        m.ops,
                        m.get_named_depthwise_bn(prefix=name).items()):
                    hidden_channel = bn.weight.numel()
                    weights.append('{}.weight'.format(bn_name))
                    penalties.append(1)
                    per_channel_flops.append(op.n_macs / hidden_channel)
    elif bn_prune_filter is None:
        # do nothing
        weights, penalties = [], []
        per_channel_flops = []
    else:
        raise NotImplementedError()

    prune_info = PruneInfo(weights, penalties)
    prune_info.add_info_list('per_channel_flops', per_channel_flops)

    if verbose:
        for name, penal in zip(prune_info.weight, prune_info.penalty):
            logging.info('{} penalty: {}'.format(name, penal))

    all_params_keys = [key for key, val in model.named_parameters()]
    for name_weight in prune_info.weight:
        assert name_weight in all_params_keys
    return prune_info


def cal_bn_l1_loss(bn_weights, penalties, rho):
    """Calculate l1 loss."""
    assert len(bn_weights) == len(penalties)
    loss = 0.0
    for weight, penal in zip(bn_weights, penalties):
        loss += rho * penal * weight.abs().sum()
    return loss


def cal_mask_network_slimming_by_flops(weights,
                                       prune_info,
                                       flops_to_prune,
                                       incremental=False):
    """Calculate mask alive atomic blocks given FLOPS target."""
    bn_weights_abs = [weight.detach().abs() for weight in weights]
    weights = torch.cat([weight for weight in bn_weights_abs])
    weights_sorted, indices = torch.sort(weights)
    flops = torch.cat([
        torch.full_like(weight, per_channel_flop)
        for weight, per_channel_flop in zip(
            bn_weights_abs, prune_info.get_info_list('per_channel_flops'))
    ])[indices]
    idx_threshold = torch.nonzero(
        torch.cumsum(flops, 0) > flops_to_prune)[0].item()
    threshold = weights_sorted[idx_threshold].item()
    mask = [weight > threshold for weight in bn_weights_abs]
    return mask, threshold


def cal_mask_network_slimming_by_threshold(weights, threshold):
    """Calculate mask alive atomic blocks given threshold."""
    bn_weights_abs = [weight.detach().abs() for weight in weights]
    weights = torch.cat(bn_weights_abs)
    mask = [weight > threshold for weight in bn_weights_abs]
    return mask


def cal_pruned_flops(prune_info):
    """Calculate total FLOPS for dead atomic blocks."""
    info = []
    pruned_flops = 0
    for name, per_channel_flops, mask in zip(
            prune_info.weight, prune_info.get_info_list('per_channel_flops'),
            prune_info.get_info_list('mask')):
        num_pruned = (~mask.detach()).sum().item()
        num_total = mask.numel()
        info.append([
            name, num_total, num_pruned, num_total * per_channel_flops,
            num_pruned * per_channel_flops, num_pruned / num_total
        ])
        pruned_flops += num_pruned * per_channel_flops
    return pruned_flops, info


def get_rho_scheduler(prune_params, steps_per_epoch):
    """Get scheduler for l1 penalty weight term.

    Only `linear` scheduler is supported.
    Support the following schedulers:
        'Linear': Could be stepwise or not.
            [0, `epoch_free`]: no regularization.
            [`epoch_free`, `epoch_warmup`]: linearly increase from 0 to `rho`.
            [`epoch_warmup`, final]: `rho`.
    """
    free_iterations = prune_params['epoch_free'] * steps_per_epoch
    warmup_iterations = prune_params['epoch_warmup'] * steps_per_epoch
    scheduler = prune_params['scheduler']
    rho = prune_params['rho']
    stepwise = prune_params['stepwise']

    def linear_fun(i):
        if not stepwise:
            i = (i // steps_per_epoch) * steps_per_epoch
        if i < free_iterations:
            return 0.0
        elif i >= warmup_iterations:
            return rho
        else:
            return (i - free_iterations) / (warmup_iterations -
                                            free_iterations) * rho

    if scheduler == 'linear':
        return linear_fun
    else:
        raise ValueError('Unknown sparsity scheduler {}'.format(scheduler))


def output_searched_network(model, infos, flags):
    """Output yaml config for searched network without dead atomic blocks."""
    inverted_residual_setting = model.inverted_residual_setting
    blocks = list(model.get_named_block_list().values())
    model_kwargs = {
        key: getattr(model, key) for key in [
            'input_channel', 'last_channel', 'width_mult', 'round_nearest',
            'active_fn', 'num_classes'
        ]
    }
    bn_prune_filter = flags.get('bn_prune_filter', None)

    res = []
    if 'skip_expand1' in bn_prune_filter:
        t, c, n, s, ks = inverted_residual_setting[0]
        assert t == 1
        assert n == 1
        assert len(ks) == 1 and ks[0] == 3
        res.append([c, n, s, ks, [model_kwargs['input_channel']], False])
        inverted_residual_setting = inverted_residual_setting[1:]
        blocks = blocks[n:]

    idx_info = 0
    for block in blocks:
        channels = []
        for k, c in zip(block.kernel_sizes, block.channels):
            info = infos[idx_info]
            assert c == info[1], '{}, {}, {}, {}'.format(block, k, c, str(info))
            num_remain = info[1] - info[2]
            channels.append(num_remain)
            idx_info += 1
        mask = [c != 0 for c in channels]
        ks_local, channels = [
            list(itertools.compress(ll, mask))
            for ll in [block.kernel_sizes, channels]
        ]
        res.append([
            block.output_dim, 1, block.stride, ks_local, channels, block.expand
        ])
    assert idx_info == len(infos)
    model_kwargs['inverted_residual_setting'] = res
    return model_kwargs
