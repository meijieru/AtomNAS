import unittest
import numpy as np
import torch

from utils.test import assertAllClose
from utils import prune


def update_bn_network_slimming(bn_weights, penalties, rho):
    """Network Slimming gradient update.

    Additional subgradient descent on the sparsity-induced penalty term.
    From paper [Learning Efficient Convolutional Networks Through Network Slimming](http://openaccess.thecvf.com/content_iccv_2017/html/Liu_Learning_Efficient_Convolutional_ICCV_2017_paper.html).
    """
    assert len(bn_weights) == len(penalties)
    for weight, penal in zip(bn_weights, penalties):
        # L1 subgradient
        weight.grad.data.add_(rho * penal * torch.sign(weight.detach()))


class PruneTest(unittest.TestCase):

    def testBnL1Loss(self):
        rho = 1.0
        penalties = [1.0]
        for i in range(10):
            var = torch.rand(10, requires_grad=True)
            var.grad = torch.zeros_like(var)
            update_bn_network_slimming([var], penalties, rho)
            lhs = var.grad.detach().cpu().numpy()

            var.grad.zero_()
            loss = prune.cal_bn_l1_loss([var], penalties, rho)
            loss.backward()
            rhs = var.grad.detach().cpu().numpy()
            assertAllClose(lhs, rhs)

    def testRhoScheduler(self):
        prune_params = {
            'rho': 1.0,
            'epoch_free': 1,
            'epoch_warmup': 3,
            'scheduler': 'linear',
            'stepwise': True,
        }
        rho_scheduler = prune.get_rho_scheduler(prune_params, 2)
        res = [rho_scheduler(i) for i in range(15)]
        expected = [0, 0, 0, 0.25, 0.50, 0.75] + [1.0] * 9
        assertAllClose(expected, res)


class MaskCalTest(unittest.TestCase):

    def testCalMaskNetworkSlimmingByThreshold(self):
        x = [
            torch.tensor(val, dtype=torch.float32)
            for val in [[1, 2, 5], [3, 6, 0, 1.1]]
        ]
        mask = prune.cal_mask_network_slimming_by_threshold(x, 1.5)
        expected = [
            torch.tensor([False, True, True]),
            torch.tensor([True, True, False, False])
        ]
        assertAllClose(mask, expected)

    def testCalMaskNetworkSlimmingByFlops(self):
        names = ['one', 'two']
        x = [
            torch.tensor(val, dtype=torch.float32)
            for val in [[1, 2, 5], [3, 6, 0, 1.1]]
        ]
        per_channel_flops = [3, 5]
        prune_info = prune.PruneInfo(names, [0, 1])
        prune_info.add_info_list('per_channel_flops', per_channel_flops)
        flops_total = sum(
            flops * len(val) for flops, val in zip(per_channel_flops, x))

        flops_to_prune = 12
        mask, threshold = prune.cal_mask_network_slimming_by_flops(
            x, prune_info, flops_to_prune)
        prune_info.add_info_list('mask', mask)
        assertAllClose(threshold, 1.1)
        expected = [
            torch.tensor([False, True, True]),
            torch.tensor([True, True, False, False])
        ]
        assertAllClose(mask, expected)
        pruned_flops, info = prune.cal_pruned_flops(prune_info)
        self.assertTrue(pruned_flops >= flops_to_prune)

        flops_to_prune = 13
        mask, threshold = prune.cal_mask_network_slimming_by_flops(
            x, prune_info, flops_to_prune)
        prune_info.add_info_list('mask', mask)
        assertAllClose(threshold, 2)
        expected = [
            torch.tensor([False, False, True]),
            torch.tensor([True, True, False, False])
        ]
        assertAllClose(mask, expected)
        pruned_flops, info = prune.cal_pruned_flops(prune_info)
        self.assertTrue(pruned_flops >= flops_to_prune)


if __name__ == "__main__":
    unittest.main()
