import copy
import unittest
import torch

from utils.test import apply_gradients
from utils.test import assertAllClose
from utils.common import check_tensor_in

from utils.rmsprop import RMSprop


class RmsPropTest(unittest.TestCase):

    def _construct_info(self):
        parameters = [torch.randn(num) for num in [5, 3, 4]]
        apply_gradients([torch.randn(num) for num in [5, 3, 4]], parameters)
        info = {
            'var_old_name': 'var_prune',
            'var_new_name': 'var_new',
            'var_new': torch.zeros(3),
            'var_old': parameters[0],
            'mask': torch.tensor([False, True, False, True, True]),
            'mask_hook': lambda lhs, rhs, mask: lhs.data.copy_(rhs.data[mask])
        }
        return parameters, info

    def testCompressWithState(self):
        parameters, info = self._construct_info()
        optimizer = RMSprop(parameters)
        optimizer.step()
        optimizer.compress_mask(info, verbose=False)
        self.assertTrue(check_tensor_in(info['var_new'], optimizer.state))
        self.assertTrue(
            check_tensor_in(info['var_new'],
                            optimizer.param_groups[0]['params']))
        self.assertFalse(check_tensor_in(parameters[0], optimizer.state))
        self.assertFalse(
            check_tensor_in(parameters[0], optimizer.param_groups[0]['params']))

    def testCompressWithoutState(self):
        parameters, info = self._construct_info()
        optimizer = RMSprop(parameters)
        optimizer.compress_mask(info, verbose=False)
        self.assertFalse(check_tensor_in(info['var_new'], optimizer.state))
        self.assertTrue(
            check_tensor_in(info['var_new'],
                            optimizer.param_groups[0]['params']))
        self.assertFalse(check_tensor_in(parameters[0], optimizer.state))
        self.assertFalse(
            check_tensor_in(parameters[0], optimizer.param_groups[0]['params']))

    def testCompressUpdate(self):
        params, info = self._construct_info()

        params0 = copy.deepcopy(params)
        apply_gradients([p.grad for p in params], params0)
        optimizer = RMSprop(params0, lr=0.1, momentum=0.5)
        optimizer.step()

        params1 = copy.deepcopy(params)
        apply_gradients([p.grad for p in params], params1)
        optimizer1 = RMSprop(params1, lr=0.1, momentum=0.5)
        optimizer1.step()

        assertAllClose(params0[1], params1[1])
        assertAllClose(params0[2], params1[2])
        assertAllClose(params0[0], params1[0])

        info['var_old'] = params1[0]
        optimizer1.compress_mask(info, verbose=True)
        optimizer1.compress_drop({'var_old': params1[2], 'type': 'variable'})
        info['mask_hook'](info['var_new'], info['var_old'], info['mask'])
        params1[0] = info['var_new']
        params1[0].grad = params0[0].grad.data[info['mask']]

        optimizer1.step()  # params1[2] not updated
        assertAllClose(params0[2], params1[2])

        optimizer.step()
        assertAllClose(params0[1], params1[1])
        assertAllClose(params0[0][info['mask']], params1[0])


if __name__ == "__main__":
    unittest.main()
