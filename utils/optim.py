"""Optimization related utils."""
from __future__ import division

import copy
from collections import OrderedDict
import logging
import functools
import importlib
import warnings
import torch
from torch import nn
from utils.rmsprop import RMSprop


class ExponentialMovingAverage(nn.Module):
    """Implement tf.train.ExponentialMovingAverage.

    TODO: implement `zero_debias=True`
    """

    def __init__(self, momentum, zero_debias=False):
        if zero_debias:
            raise NotImplementedError('zero_debias')
        if momentum < 0.0 or momentum > 1.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        super(ExponentialMovingAverage, self).__init__()
        self._momentum = momentum
        self._zero_debias = zero_debias
        self.clear()

    def _check_exist(self, name):
        if name not in self._shadow:
            raise RuntimeError('{} has not been registered'.format(name))

    def register(self, name, val, zero_init=False):
        """Register and init variable to averaged."""
        if name in self._shadow:
            raise ValueError('Should not register twice for {}'.format(name))
        if val.dtype not in [torch.float16, torch.float32, torch.float64]:
            raise TypeError(
                'The variables must be half, float, or double: {}'.format(name))

        if zero_init:
            self._shadow[name] = torch.zeros_like(val)
        else:
            self._shadow[name] = val.detach().clone()
        self._info[name] = {
            'num_updates': 0,
            'last_momemtum': None,
            'zero_init': zero_init,
            'compress_masked': False,
        }

    def forward(self, name, x, num_updates=None):
        """Update averaged variable."""
        self._check_exist(name)
        if num_updates is None:
            momentum = self._momentum
        else:
            momentum = min(self._momentum,
                           (1.0 + num_updates) / (10.0 + num_updates))
        self._info[name]['num_updates'] += 1
        self._info[name]['last_momemtum'] = momentum
        return self._shadow[name].mul_(momentum).add_(1.0 - momentum,
                                                      x.detach())

    def clear(self):
        """Remove all registered variables."""
        self._shadow = OrderedDict()
        self._info = OrderedDict()

    def pop(self, name):
        """Remove and return info."""
        self._check_exist(name)
        val = self._shadow.pop(name)
        info = self._info.pop(name)
        return val, info

    def average_names(self):
        """Get names of all registered variables."""
        return list(self._shadow.keys())

    def average(self, name):
        """Get averaged variable."""
        self._check_exist(name)
        return self._shadow[name]

    def state_dict(self):
        return {
            'info': self._info,
            'shadow': self._shadow,
            'param': {
                'momentum': self._momentum,
                'zero_debias': self._zero_debias
            }
        }

    def load_state_dict(self, state_dict):
        params = state_dict['param']
        for key, val in params.items():
            cur_val = getattr(self, '_{}'.format(key))
            if val != cur_val:
                warning_str = 'EMA {} mismatch: current {} vs previous {}'.format(
                    key, cur_val, val)
                warnings.warn(warning_str, RuntimeWarning)
                logging.warning(warning_str)
        self._shadow = copy.deepcopy(state_dict['shadow'])
        self._info = copy.deepcopy(state_dict['info'])

    def to(self, *args, **kwargs):
        device, dtype, non_blocking = torch._C._nn._parse_to(*args, **kwargs)
        for k in list(self._shadow.keys()):
            v = self._shadow[k]
            self._shadow[k] = v.to(device,
                                   dtype if v.is_floating_point() else None,
                                   non_blocking)
        return self

    def compress_start(self):
        """Setup info used for dynamic network shrinkage.

        Typical dynamic network shrinkage for EMA:
            ```
            ema.compress_start()
            for info in infos_mask:
                ema.compress_mask(*args, **kwargs)
            for info in infos_drop:
                ema.compress_drop(*args, **kwargs)
            ```
        """
        for val in self._info.values():
            val['compress_masked'] = False

    def compress_mask(self, info, verbose=False):
        """Adjust parameters values by masks for dynamic network shrinkage."""
        var_old_name = info['var_old_name']
        var_new_name = info['var_new_name']
        var_new = info['var_new']
        mask_hook = info['mask_hook']
        mask = info['mask']

        if verbose:
            logging.info('EMA compress: {} -> {}'.format(var_old_name, var_new_name))
        if self._info[var_old_name]['compress_masked']:
            raise RuntimeError('May have dependencies in compress')
        if var_new_name in self._info and self._info[var_new_name]['compress_masked']:
            raise RuntimeError('Compress {} twice'.format(var_new_name))
        ema_old = self._shadow.pop(var_old_name)
        ema_new = torch.zeros_like(var_new, device=ema_old.device)
        mask_hook(ema_new, ema_old, mask)
        self._info[var_new_name] = self._info.pop(var_old_name)
        self._info[var_new_name]['compress_masked'] = True
        self._shadow[var_new_name] = ema_new

    def compress_drop(self, info, verbose=False):
        """Remove unused parameters for dynamic network shrinkage."""
        name = info['var_old_name']
        if verbose:
            logging.info('EMA drop: {}'.format(name))
        self._check_exist(name)
        if self._info[name]['compress_masked']:
            if verbose:
                logging.info('EMA drop: {} skipped'.format(name))
        else:
            return self.pop(name)

    @staticmethod
    def adjust_momentum(momentum, steps_multi):
        """Adjust EMA's momentum according to number of steps.

        With different batch size, the number of steps varies. So the momentum
        should be adjusted to minic the performance base steps. This strategy
        assumes checkpoint within certain time steps are similar.
        NOTE: There is still no guarantee about the performance when use
            different batch size.
        """
        return momentum**(1.0 / steps_multi)


class CrossEntropyLabelSmooth(nn.Module):
    """Label smoothed version of CrossEntropy"""

    def __init__(self, num_classes, label_smoothing, reduction='none'):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.label_smoothing = label_smoothing
        self.logsoftmax = nn.LogSoftmax(dim=1)

        if reduction == 'none':
            fun = lambda x: x
        elif reduction == 'mean':
            fun = torch.mean
        elif reduction == 'sum':
            fun = torch.sum
        else:
            raise ValueError('Unknown reduction: {}'.format(reduction))
        self.reduce_fun = fun

    def forward(self, inputs, targets):
        assert inputs.size(1) == self.num_classes
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1),
                                                       1)
        targets = (1 - self.label_smoothing
                  ) * targets + self.label_smoothing / self.num_classes
        loss = torch.sum(-targets * log_probs, 1)
        return self.reduce_fun(loss)


def cal_l2_loss(model, weight_decay, method):
    """Calculate l2 penalty."""
    # TODO: profile performance
    loss = 0.0
    if method == 'slimmable':
        for params in model.parameters():
            # all depthwise convolution (N, 1, x, x) has no weight decay
            # weight decay only on normal conv and fc
            ps = list(params.size())
            if len(ps) == 4 and ps[1] != 1:
                _weight_decay = weight_decay
            elif len(ps) == 2:
                _weight_decay = weight_decay
            else:
                _weight_decay = 0
            loss += _weight_decay * (params**2).sum()
    elif method == 'mnas':
        classifier_bias_count = 0
        weight_decay_map = dict()
        for name, params in model.named_parameters():
            ps = list(params.size())
            if len(ps) == 4 or len(ps) == 2:
                # regularize all convolution and fc weight
                weight_decay_map[name] = weight_decay, params
            else:
                assert len(ps) == 1
                if 'classifier' in name:  # fc bias
                    weight_decay_map[name] = weight_decay, params
                    classifier_bias_count += 1
                else:  # bn weight/bias
                    weight_decay_map[name] = 0.0, params
        assert classifier_bias_count == 1
        for _weight_decay, params in weight_decay_map.values():
            loss += _weight_decay * (params**2).sum()
    elif method == 'mnas_no_bias':
        # TODO: mnas_no_bias
        raise NotImplementedError()
    else:
        raise ValueError('Unknown weight_decay method: {}'.format(method))
    return loss * 0.5


def get_lr_scheduler(optimizer, FLAGS):
    """Get learning rate scheduler."""
    stepwise = FLAGS.get('lr_stepwise', True)
    steps_per_epoch = FLAGS._steps_per_epoch
    warmup_iterations = FLAGS.get('epoch_warmup', 5) * steps_per_epoch
    use_warmup = FLAGS.lr > FLAGS.base_lr

    # TODO: warmup rewrite
    # warmup
    def warmup_wrap(lr_lambda, i):
        if use_warmup and i <= warmup_iterations:
            warmup_lr_ratio = FLAGS.base_lr / FLAGS.lr
            return warmup_lr_ratio + i / warmup_iterations * (1 -
                                                              warmup_lr_ratio)
        else:
            return lr_lambda(i)

    if FLAGS.lr_scheduler == 'multistep':
        if use_warmup:
            raise NotImplementedError('Warmup not implemented for multistep')
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[
                steps_per_epoch * val for val in FLAGS.multistep_lr_milestones
            ],
            gamma=FLAGS.multistep_lr_gamma)
    elif FLAGS.lr_scheduler == 'exp_decaying' or FLAGS.lr_scheduler == 'exp_decaying_trunc':

        def aux(i, trunc_to_constant=0.0):
            decay_interval = steps_per_epoch * FLAGS.exp_decay_epoch_interval
            if not stepwise:
                i = (i // decay_interval) * decay_interval
            res = FLAGS.exp_decaying_lr_gamma**(i / decay_interval)
            return res if res > trunc_to_constant else trunc_to_constant

        if 'trunc' in FLAGS.lr_scheduler:
            trunc_to_constant = 0.05
        else:
            trunc_to_constant = 0.0
        lr_lambda = functools.partial(
            warmup_wrap,
            functools.partial(aux, trunc_to_constant=trunc_to_constant))
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                         lr_lambda=lr_lambda)
    elif FLAGS.lr_scheduler == 'linear_decaying':
        assert stepwise
        lr_lambda = functools.partial(
            warmup_wrap, lambda i: 1 - i / (FLAGS.num_epochs * steps_per_epoch))
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                         lr_lambda=lr_lambda)
    else:
        raise NotImplementedError(
            'Learning rate scheduler {} is not yet implemented.'.format(
                FLAGS.lr_scheduler))
    return lr_scheduler


def get_optimizer(model, FLAGS):
    """Get optimizer."""
    if FLAGS.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=FLAGS.lr,
                                    momentum=FLAGS.momentum,
                                    nesterov=FLAGS.nesterov,
                                    weight_decay=0)
    elif FLAGS.optimizer == 'rmsprop':
        optimizer = RMSprop(model.parameters(),
                            lr=FLAGS.lr,
                            alpha=FLAGS.alpha,
                            momentum=FLAGS.momentum,
                            eps=FLAGS.epsilon,
                            eps_inside_sqrt=FLAGS.eps_inside_sqrt,
                            weight_decay=0)
    else:
        try:
            optimizer_lib = importlib.import_module(FLAGS.optimizer)
            return optimizer_lib.get_optimizer(model)
        except ImportError:
            raise NotImplementedError(
                'Optimizer {} is not yet implemented.'.format(FLAGS.optimizer))
    return optimizer
