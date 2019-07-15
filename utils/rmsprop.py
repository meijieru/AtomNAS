"""Customized implementation of RMSprop."""
import logging
import torch
from torch.optim.optimizer import Optimizer
from utils.common import index_tensor_in
from utils.common import check_tensor_in


class RMSprop(Optimizer):
    """Implements RMSprop algorithm.

    Proposed by G. Hinton in his
    `course <http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf>`_.

    The centered version first appears in `Generating Sequences
    With Recurrent Neural Networks <https://arxiv.org/pdf/1308.0850v5.pdf>`_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-2)
        momentum (float, optional): momentum factor (default: 0)
        alpha (float, optional): smoothing constant (default: 0.99)
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        eps_inside_sqrt (float, optional): if ``True``, add eps inside the sqrt.
            (default: False)
        centered (bool, optional) : if ``True``, compute the centered RMSProp,
            the gradient is normalized by an estimation of its variance
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)

    """

    def __init__(self,
                 params,
                 lr=1e-2,
                 alpha=0.99,
                 eps=1e-8,
                 eps_inside_sqrt=False,
                 weight_decay=0,
                 momentum=0,
                 centered=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= momentum:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if not 0.0 <= weight_decay:
            raise ValueError(
                "Invalid weight_decay value: {}".format(weight_decay))
        if not 0.0 <= alpha:
            raise ValueError("Invalid alpha value: {}".format(alpha))

        defaults = dict(lr=lr,
                        momentum=momentum,
                        alpha=alpha,
                        eps=eps,
                        eps_inside_sqrt=eps_inside_sqrt,
                        centered=centered,
                        weight_decay=weight_decay)
        super(RMSprop, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(RMSprop, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('momentum', 0)
            group.setdefault('centered', False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        'RMSprop does not support sparse gradients')
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['square_avg'] = torch.zeros_like(p.data)
                    if group['momentum'] > 0:
                        state['momentum_buffer'] = torch.zeros_like(p.data)
                    if group['centered']:
                        state['grad_avg'] = torch.zeros_like(p.data)

                square_avg = state['square_avg']
                alpha = group['alpha']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                square_avg.mul_(alpha).addcmul_(1 - alpha, grad, grad)

                if group['centered']:
                    grad_avg = state['grad_avg']
                    grad_avg.mul_(alpha).add_(1 - alpha, grad)
                    if group['eps_inside_sqrt']:
                        avg = square_avg.addcmul(-1, grad_avg, grad_avg).add_(
                            group['eps']).sqrt_()
                    else:
                        avg = square_avg.addcmul(
                            -1, grad_avg, grad_avg).sqrt_().add_(group['eps'])
                else:
                    if group['eps_inside_sqrt']:
                        avg = square_avg.add(group['eps']).sqrt_()
                    else:
                        avg = square_avg.sqrt().add_(group['eps'])

                if group['momentum'] > 0:
                    buf = state['momentum_buffer']
                    buf.mul_(group['momentum']).addcdiv_(grad, avg)
                    p.data.add_(-group['lr'], buf)
                else:
                    p.data.addcdiv_(-group['lr'], grad, avg)

        return loss

    def compress_mask(self, info, verbose=False):
        """Adjust parameters values by masks for dynamic network shrinkage."""
        var_old = info['var_old']
        var_new = info['var_new']
        mask_hook = info['mask_hook']
        mask = info['mask']
        if verbose:
            logging.info('RMSProp compress: {} -> {}'.format(
                info['var_old_name'], info['var_new_name']))

        found = False
        for group in self.param_groups:
            index = index_tensor_in(var_old, group['params'], raise_error=False)
            found = index is not None
            if found:
                if check_tensor_in(var_old, self.state):
                    state = self.state.pop(var_old)
                    if len(state) != 0:  # generate new state
                        new_state = {'step': state['step']}
                        for key in ['square_avg', 'momentum_buffer', 'grad_avg']:
                            if key in state:
                                new_state[key] = torch.zeros_like(
                                    var_new.data, device=var_old.device)
                                mask_hook(new_state[key], state[key], mask)
                                new_state[key].to(state[key].device)
                        self.state[var_new] = new_state

                # update group
                del group['params'][index]
                group['params'].append(var_new)
                break
        assert found, 'Var: {} not in RMSProp'.format(info['var_old_name'])

    def compress_drop(self, info, verbose=False):
        """Remove unused parameters for dynamic network shrinkage."""
        var_old = info['var_old']
        if verbose:
            logging.info('RMSProp drop: {}'.format(info['var_old_name']))

        assert info['type'] == 'variable'
        found = False
        for group in self.param_groups:
            index = index_tensor_in(var_old, group['params'], raise_error=False)
            found = index is not None
            if found:
                if check_tensor_in(var_old, self.state):
                    self.state.pop(var_old)
                del group['params'][index]
        assert found, 'Var: {} not in RMSProp'.format(info['var_old_name'])
