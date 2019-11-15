import unittest
import easydict
import numpy as np
import torch

from utils import optim
from utils.test import assertAllClose
from utils.test import to_numpy


class CrossEntropyLabelSmoothTest(unittest.TestCase):
    """Port from tensorflow"""

    def testSoftmaxLabelSmoothing(self):
        # Softmax Cross Entropy Loss is:
        #   -\sum_i p_i \log q_i
        # where for a softmax activation
        # \log q_i = x_i - \log \sum_j \exp x_j
        #          = x_i - x_max - \log \sum_j \exp (x_j - x_max)
        # For our activations, [100, -100, -100] the log partition function
        # becomes \log ( exp(0) + exp(-200) + exp(-200) ) = 0
        # so our log softmaxes become: [0, -200, -200]
        # so our cross entropy loss is:
        # -(1 - L + L/n) * 0 + 400 * L/n = 400 L/n
        logits = torch.tensor([[100.0, -100.0, -100.0]])
        labels = torch.tensor([0], dtype=torch.int64)
        label_smoothing = 0.1
        criterion = optim.CrossEntropyLabelSmooth(logits.size(1),
                                                  label_smoothing)
        expected_value = 400.0 * label_smoothing / 3.0
        res = criterion(logits, labels).item()
        assertAllClose(res, expected_value)


def _Repeat(value, dim):
    if dim == 1:
        return value
    return [value] * dim


class ExponentialMovingAverageTest(unittest.TestCase):
    """Port from tensorflow"""

    def _CheckDecay(self,
                    ema,
                    actual_decay,
                    dim,
                    num_updates=None,
                    vars_pre_hooks=None,
                    num_updates_post_hook=None):

        def _Update():
            nonlocal num_updates
            if vars_pre_hooks is not None:
                assert len(vals) == vars_pre_hooks
                for val, var_prehook in zip(vals, vars_pre_hooks):
                    var_prehook(val)
            for name, val in zip(names, vals):
                ema(name, val, num_updates)
            if num_updates_post_hook:
                num_updates = num_updates_post_hook

        def _Scale(dk, steps):
            if ema._zero_debias:
                return 1 - dk**steps
            else:
                return 1

        tens = _Repeat(10.0, dim)
        thirties = _Repeat(30.0, dim)
        var0 = torch.tensor(tens)
        var1 = torch.tensor(thirties)
        # Note that tensor2 is not a Variable but just a plain Tensor resulting
        # from the sum operation.
        tensor2 = var0 + var1
        names = ['tens', 'thirties', 'tensor2']
        vals = [var0, var1, tensor2]
        zero_inits = [False, False, True]
        for name, var, zero_init in zip(names, vals, zero_inits):
            ema.register(name, var, zero_init)

        # Check that averages are initialized correctly.
        assertAllClose(tens, ema.average('tens'))
        assertAllClose(thirties, ema.average('thirties'))
        # Note that averages of Tensor's initialize to zeros_like since no value
        # of the Tensor is known because the Op has not been run (yet).
        assertAllClose(_Repeat(0.0, dim), ema.average('tensor2'))

        # Update the averages and check.
        _Update()
        dk = actual_decay

        expected = _Repeat(10.0 * dk + 10.0 * (1 - dk), dim)
        assertAllClose(expected, ema.average('tens'))
        expected = _Repeat(30.0 * dk + 30.0 * (1 - dk), dim)
        assertAllClose(expected, ema.average('thirties'))
        expected = _Repeat(0.0 * dk + (10.0 + 30.0) * (1 - dk) / _Scale(dk, 1),
                           dim)
        assertAllClose(expected, ema.average('tensor2'))

        # Again, update the averages and check.
        _Update()
        expected = _Repeat((10.0 * dk + 10.0 * (1 - dk)) * dk + 10.0 * (1 - dk),
                           dim)
        assertAllClose(expected, ema.average('tens'))
        expected = _Repeat((30.0 * dk + 30.0 * (1 - dk)) * dk + 30.0 * (1 - dk),
                           dim)
        assertAllClose(expected, ema.average('thirties'))
        expected = _Repeat(((0.0 * dk + (10.0 + 30.0) * (1 - dk)) * dk +
                            (10.0 + 30.0) * (1 - dk)) / _Scale(dk, 2), dim)
        assertAllClose(expected, ema.average('tensor2'))

    def testAverageVariablesNoNumUpdates_Scalar(self):
        ema = optim.ExponentialMovingAverage(0.25)
        self._CheckDecay(ema, actual_decay=0.25, dim=1)

    def testAverageVariablesNoNumUpdates_Vector(self):
        ema = optim.ExponentialMovingAverage(0.25)
        self._CheckDecay(ema, actual_decay=0.25, dim=5)

    def testAverageVariablesNumUpdates_Scalar(self):
        # With num_updates 1, the decay applied is 0.1818
        ema = optim.ExponentialMovingAverage(0.25)
        self._CheckDecay(ema, actual_decay=0.181818, dim=1, num_updates=1)

    def testAverageVariablesNumUpdates_Vector(self):
        ema = optim.ExponentialMovingAverage(0.25)
        self._CheckDecay(ema, actual_decay=0.181818, dim=5, num_updates=1)

    @unittest.skip('zero_debias not implemented')
    def testAverageVariablesNoNumUpdates_Scalar_Debias(self):
        ema = optim.ExponentialMovingAverage(0.25, zero_debias=True)
        self._CheckDecay(ema, actual_decay=0.25, dim=1)

    @unittest.skip('zero_debias not implemented')
    def testAverageVariablesNoNumUpdates_Vector_Debias(self):
        ema = optim.ExponentialMovingAverage(0.25, zero_debias=True)
        self._CheckDecay(ema, actual_decay=0.25, dim=5)

    @unittest.skip('zero_debias not implemented')
    def testAverageVariablesNumUpdates_Scalar_Debias(self):
        # With num_updates 1, the decay applied is 0.1818
        ema = optim.ExponentialMovingAverage(0.25, zero_debias=True)
        self._CheckDecay(ema, actual_decay=0.181818, dim=1, num_updates=1)

    @unittest.skip('zero_debias not implemented')
    def testAverageVariablesNumUpdates_Vector_Debias(self):
        # With num_updates 1, the decay applied is 0.1818
        ema = optim.ExponentialMovingAverage(0.25, zero_debias=True)
        self._CheckDecay(ema, actual_decay=0.181818, dim=5, num_updates=1)

    def testThrowValueError(self):
        ema = optim.ExponentialMovingAverage(0.25)
        try:
            ema.register('test', torch.tensor(5, dtype=torch.int))
        except TypeError:
            pass
        else:
            raise AssertionError('Should throw ValueError')

    def testAverageVariablesUpdateNumUpdates_Vector(self):
        ema = optim.ExponentialMovingAverage(0.25)
        name = 'tens'
        tens = _Repeat(10.0, dim=5)
        var = torch.tensor(tens)
        ema.register(name, var, zero_init=False)
        for num_updates in range(2):
            var.add_(1)
            ema(name, var, num_updates=num_updates)
        expected = _Repeat((10 * 0.1 + 11 * 0.9) * 2.0 / 11.0 + 12 * 9.0 / 11.0,
                           dim=5)
        assertAllClose(expected, ema.average(name))

    def testSaveLoad(self):
        ema = optim.ExponentialMovingAverage(0.25)
        name = 'tens'
        tens = _Repeat(10.0, dim=5)
        var = torch.tensor(tens)
        ema.register(name, var, zero_init=False)
        state_dict = ema.state_dict()
        for name in ['info', 'shadow', 'param']:
            assert name in state_dict
        assert 'tens' in state_dict['shadow']
        assertAllClose(state_dict['shadow']['tens'], var)

        ema.load_state_dict(state_dict)
        state_dict['param']['momentum'] = 0.5
        self.assertWarns(RuntimeWarning,
                         lambda: ema.load_state_dict(state_dict))

    def testCompress(self):
        ema = optim.ExponentialMovingAverage(0.25)
        ema.register('var_prune', torch.arange(5).float())
        ema.register('var_keep', torch.arange(5, 10).float())
        ema('var_prune', torch.arange(5).float())
        info = {
            'var_old_name': 'var_prune',
            'var_new_name': 'var_new',
            'var_new': torch.randn(3),
            'mask': torch.tensor([False, True, False, True, True]),
            'mask_hook': lambda lhs, rhs, mask: lhs.data.copy_(rhs.data[mask])
        }
        ema.compress_mask(info, verbose=False)
        self.assertTrue(info['var_new_name'] in ema._shadow)
        self.assertTrue(info['var_new_name'] in ema._info)
        self.assertTrue(info['var_old_name'] not in ema._shadow)
        self.assertTrue(info['var_old_name'] not in ema._info)
        self.assertEqual(ema._info[info['var_new_name']]['num_updates'], 1)
        assertAllClose(ema.average(info['var_new_name']), [1, 3, 4])

    def testAdjustEmaRate(self):
        num_repeat = 3
        for num_repeat in [1, 5, 9]:
            for momentum in [0.25, 0.9999]:
                momentum = 0.25
                name = 'v'
                values = torch.randn(5)
                values_long = values.repeat(num_repeat, 1).permute(
                    (1, 0)).contiguous().view(-1)

                ema = optim.ExponentialMovingAverage(momentum)
                ema.register(name, values[0])
                for v in values:
                    ema(name, v)
                lhs = ema.average(name)

                momentum = optim.ExponentialMovingAverage.adjust_momentum(
                    momentum, num_repeat)
                ema = optim.ExponentialMovingAverage(momentum)
                ema.register(name, values[0])
                for v in values_long:
                    ema(name, v)
                rhs = ema.average(name)

                assertAllClose(lhs, rhs)

    @unittest.skipIf(not torch.cuda.is_available(), 'GPU test not available')
    def testDevice(self):
        name = 'zeros'
        ema = optim.ExponentialMovingAverage(0.25)
        ema.register(name, torch.zeros(5))
        for device in ['cpu', 'cuda:0']:
            device = torch.device(device)
            ema = ema.to(device)
            cur_device = ema._shadow[name].device
            assert cur_device == device


class WeightDecayTest(unittest.TestCase):

    def testSgdDecay(self):
        var = np.random.randn(10)
        weight_decay = 1e-1

        var0 = torch.tensor(var, requires_grad=True, dtype=torch.float32)
        var0.grad = torch.zeros_like(var0)
        optimizer = torch.optim.SGD([var0], weight_decay=weight_decay, lr=0.1)
        optimizer.zero_grad()
        optimizer.step()

        var1 = torch.tensor(var, requires_grad=True, dtype=torch.float32)
        optimizer = torch.optim.SGD([var1], weight_decay=0, lr=0.1)
        optimizer.zero_grad()
        loss = (weight_decay * 0.5) * (var1**2).sum()
        loss.backward()
        optimizer.step()

        assertAllClose(to_numpy(var0.grad), to_numpy(var1.grad))
        assertAllClose(to_numpy(var0), to_numpy(var1))


class LrSchedulerTest(unittest.TestCase):

    def _setup(self, lr=1.0):
        var = torch.randn(5)
        optimizer = torch.optim.SGD([var], lr=lr)
        return optimizer

    def _step(self, optimizer, lr_scheduler, steps):
        res = []
        for i in range(steps):
            res.append(optimizer.param_groups[0]['lr'])
            optimizer.step()
            lr_scheduler.step()
        return res

    def testExpDecaying(self):
        exp_decaying_lr_gamma = 0.66
        FLAGS = easydict.EasyDict({
            'lr_scheduler': 'exp_decaying',
            'epoch_warmup': 5,
            '_steps_per_epoch': 1,
            'lr': 0.256,
            'base_lr': 0.256,
            'exp_decay_epoch_interval': 2,
            'exp_decaying_lr_gamma': exp_decaying_lr_gamma,
            'lr_stepwise': False,
        })
        optimizer = self._setup(FLAGS.lr)
        lr_scheduler = optim.get_lr_scheduler(optimizer, FLAGS)
        res = self._step(optimizer, lr_scheduler, 21)
        self.assertEqual(res[-1], FLAGS.lr * FLAGS.exp_decaying_lr_gamma ** 10)

        optimizer = self._setup(FLAGS.lr)
        FLAGS.lr_scheduler = 'exp_decaying_trunc'
        lr_scheduler = optim.get_lr_scheduler(optimizer, FLAGS)
        res = self._step(optimizer, lr_scheduler, 122)
        self.assertEqual(res[-1], FLAGS.lr * 0.05)


if __name__ == "__main__":
    unittest.main()
