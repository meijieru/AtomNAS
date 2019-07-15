import unittest
import torch

import models.mobilenet_base as mb


def random_bn(m):
    if isinstance(m, torch.nn.BatchNorm2d):
        for attr in ['running_mean', 'running_var', 'weight', 'bias']:
            getattr(m, attr).data.copy_(torch.rand_like(m.weight))


class InvertedResidualChannelsTest(unittest.TestCase):

    def _run_compress(self, expand):

        def compare(m):
            m.apply(random_bn)
            bn_list = m.get_depthwise_bn()
            masks = [
                torch.tensor([False, True]),
                torch.tensor([True, True, True, False])
            ]
            bn_list[0].weight[0] = 0
            bn_list[1].weight[3] = 0
            x = torch.randn(6, inp, 10, 10)
            lhs = m(x)
            m.compress_by_mask(masks)
            rhs = m(x)
            assertAllClose(lhs, rhs)

        inp, oup = 3, 5
        m = mb.InvertedResidualChannels(inp, oup, 1, [2, 4], [3, 5], expand,
                                        torch.nn.ReLU)
        m.train()
        compare(m)

        m = mb.InvertedResidualChannels(inp, oup, 1, [2, 4], [3, 5], expand,
                                        torch.nn.ReLU)
        m.eval()
        compare(m)

    def testCompressExpand(self):
        self._run_compress(True)

    def testCompressDrop(self):
        inp, oup = 3, 5
        m = mb.InvertedResidualChannels(inp, oup, 1, [2, 4], [3, 5], True,
                                        torch.nn.ReLU)
        masks = [
            torch.tensor([False, False]),
            torch.tensor([True, True, True, False])
        ]
        m.compress_by_mask(masks)
        assert len(m.ops) == 1

    @unittest.skip('Do not search when not expand')
    def testCompressNonExpand(self):
        self._run_compress(False)

    def testCompressWithEmaOptimizerPruneinfo(self):
        from utils.optim import ExponentialMovingAverage
        from utils.rmsprop import RMSprop
        from utils.prune import PruneInfo

        import sys
        import logging
        logging.basicConfig(stream=sys.stdout,
                            level=logging.INFO,
                            datefmt='%m/%d %I:%M:%S %p')

        def create_ema(m):
            ema = ExponentialMovingAverage(0.25)
            for name, param in m.named_parameters():
                ema.register(name, param)
            for name, param in m.named_buffers():
                if 'running_var' in name or 'running_mean' in name:
                    ema.register(name, param)
            return ema

        def get_weight_name(index):
            return 'ops.{}.1.1.weight'.format(index)

        inp, oup, expand = 3, 5, True
        m = mb.InvertedResidualChannels(inp, oup, 1, [2, 4], [3, 5], expand,
                                        torch.nn.ReLU)
        num_var = len(list(m.parameters()))
        optimizer = RMSprop(m.parameters())
        ema = create_ema(m)
        prune_info = PruneInfo([get_weight_name(i) for i in range(len(m.ops))],
                               [1, 2])
        masks = [
            torch.tensor([False, False]),
            torch.tensor([True, True, True, False])
        ]
        m.compress_by_mask(masks,
                           ema=ema,
                           optimizer=optimizer,
                           prune_info=prune_info,
                           verbose=False)
        ema2 = create_ema(m)
        self.assertLess(len(list(m.parameters())), num_var)
        self.assertEqual(set(m.parameters()),
                         set(optimizer.param_groups[0]['params']))
        self.assertEqual(set(ema.average_names()), set(ema2.average_names()))
        self.assertListEqual(prune_info.weight, [get_weight_name(0)])
        self.assertEqual(len(prune_info.weight), 1)
        self.assertListEqual(prune_info.penalty, [2])

    def _run_get_depthwise_bn(self, expand):
        inp, oup = 6, 5
        m = mb.InvertedResidualChannels(inp, oup, 1, [2, 4], [3, 5], expand,
                                        torch.nn.ReLU)
        bn_list = m.get_depthwise_bn()
        for bn in bn_list:
            assert isinstance(bn, torch.nn.BatchNorm2d)

    def testGetDepthwiseBnExpand(self):
        self._run_get_depthwise_bn(True)

    def testGetNamedDepthwiseBnExpand(self):
        inp, oup = 6, 5
        m = mb.InvertedResidualChannels(inp, oup, 1, [2, 4], [3, 5], True,
                                        torch.nn.ReLU)
        self.assertListEqual(list(m.get_named_depthwise_bn()),
                             ['ops.{}.1.1'.format(i) for i in range(2)])
        prefix = 'prefix'
        self.assertListEqual(
            list(m.get_named_depthwise_bn(prefix=prefix)),
            ['{}.ops.{}.1.1'.format(prefix, i) for i in range(2)])

    @unittest.skip('Do not search when not expand')
    def testGetDepthwiseBnNonExpand(self):
        self._run_get_depthwise_bn(False)


class InvertedResidualChannelsFusedTest(unittest.TestCase):

    def _run_get_depthwise_bn(self, expand):
        inp, oup = 6, 5
        m = mb.InvertedResidualChannelsFused(inp, oup, 1, [2, 4], [3, 5],
                                             expand, torch.nn.ReLU)
        bn_list = m.get_depthwise_bn()
        for bn in bn_list:
            assert isinstance(bn, torch.nn.BatchNorm2d)

    def testGetDepthwiseBnExpand(self):
        self._run_get_depthwise_bn(True)

    def testGetNamedDepthwiseBnExpand(self):
        inp, oup = 6, 5
        m = mb.InvertedResidualChannelsFused(inp, oup, 1, [2, 4], [3, 5], True,
                                             torch.nn.ReLU)
        self.assertListEqual(list(m.get_named_depthwise_bn()),
                             ['depth_ops.{}.1.1'.format(i) for i in range(2)])
        prefix = 'prefix'
        self.assertListEqual(
            list(m.get_named_depthwise_bn(prefix=prefix)),
            ['{}.depth_ops.{}.1.1'.format(prefix, i) for i in range(2)])


if __name__ == "__main__":
    unittest.main()
