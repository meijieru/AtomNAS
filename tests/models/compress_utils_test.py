import unittest
import torch
from torch import nn

from utils.test import assertAllClose
import models.mobilenet_base as mb
import models.compress_utils as cu


def random_bn(m):
    if isinstance(m, torch.nn.BatchNorm2d):
        for attr in ['running_mean', 'running_var', 'weight', 'bias']:
            getattr(m, attr).data.copy_(torch.rand_like(m.weight))


class CompressUtilsTest(unittest.TestCase):

    def _apply_info(self, infos):
        for info in infos:
            try:
                info['mask_hook'](info['var_new'], info['var_old'],
                                  info['mask'])
            except BaseException as e:
                raise e

    def assertEqualId(self, lhs, rhs):
        assert id(lhs) == id(rhs)

    def testCompressConvAttr(self):
        inp, oup, kernel_size, groups = 3, 5, 3, 1
        conv0 = nn.Conv2d(inp, oup, kernel_size, groups=groups)
        conv1 = nn.Conv2d(inp, oup, kernel_size, groups=groups)
        mask = torch.ones(oup)
        infos = cu.compress_conv(conv1,
                                 conv0,
                                 mask,
                                 0,
                                 prefix_new='new',
                                 prefix_old='old')
        for info, attr in zip(infos, ['weight', 'bias']):
            self.assertEqual(info['type'], 'variable')
            self.assertEqualId(info['var_old'], getattr(conv0, attr))
            self.assertEqualId(info['var_new'], getattr(conv1, attr))
            self.assertEqualId(info['mask'], mask)
            self.assertEqual(info['module_class'], nn.Conv2d)
            for name in ['new', 'old']:
                self.assertEqual(info['var_{}_name'.format(name)],
                                 '{}.{}'.format(name, attr))

    def testCompressConv(self):
        inp, oup, kernel_size, groups = 3, 5, 3, 1
        conv0 = nn.Conv2d(inp, oup, kernel_size, groups=groups)

        mask = torch.tensor([False, True, True, False, False])
        conv1 = nn.Conv2d(inp, mask.sum().item(), kernel_size, groups=groups)
        infos = cu.compress_conv(conv1, conv0, mask, 0)
        self._apply_info(infos)

        inputs = torch.randn(1, 3, 10, 10)
        lhs = conv0(inputs)
        rhs = conv1(inputs)
        assertAllClose(lhs[:, mask], rhs)

    def testCompressConvDim1(self):
        inp, oup, kernel_size, groups = 3, 5, 3, 1
        conv0 = nn.Conv2d(inp, oup, kernel_size, groups=groups)

        mask = torch.tensor([False, True, True])
        conv1 = nn.Conv2d(mask.sum().item(), oup, kernel_size, groups=groups)
        infos = cu.compress_conv(conv1, conv0, mask, 1)
        self._apply_info(infos)

        inputs = torch.randn(1, 3, 10, 10)
        inputs[0, 0] = 0
        lhs = conv0(inputs)
        rhs = conv1(inputs[:, mask])
        assertAllClose(lhs, rhs)

    def testCompressConvDepthwise(self):
        inp, oup, kernel_size, groups = 5, 5, 3, 5
        conv0 = nn.Conv2d(inp, oup, kernel_size, groups=groups)
        mask = torch.tensor([False, True, True, False, False])
        num_remain = mask.sum().item()
        conv1 = nn.Conv2d(num_remain,
                          num_remain,
                          kernel_size,
                          groups=num_remain)
        infos = cu.compress_conv(conv1, conv0, mask, 0)
        self._apply_info(infos)

        inputs = torch.randn(1, 5, 10, 10)
        lhs = conv0(inputs)
        rhs = conv1(inputs[:, mask])
        assertAllClose(lhs[:, mask], rhs)

    def testCompressBnAttr(self):
        inp = 5
        m0 = nn.BatchNorm2d(inp)
        m1 = nn.BatchNorm2d(inp)
        mask = torch.ones(inp)
        infos = cu.compress_bn(m1, m0, mask, prefix_new='new', prefix_old='old')
        for info, attr in zip(infos, [
                'weight', 'bias', 'running_var', 'running_mean',
                'num_batches_tracked'
        ]):
            if attr in ['weight', 'bias']:
                self.assertEqual(info['type'], 'variable')
            else:
                self.assertEqual(info['type'], 'buffer')
            self.assertEqualId(info['var_old'], getattr(m0, attr))
            self.assertEqualId(info['var_new'], getattr(m1, attr))
            self.assertEqualId(info['mask'], mask)
            self.assertEqual(info['module_class'], nn.BatchNorm2d)
            for name in ['new', 'old']:
                self.assertEqual(info['var_{}_name'.format(name)],
                                 '{}.{}'.format(name, attr))

    def testCompressConvBnRelu(self):
        inp, oup, kernel_size, groups = 3, 5, 3, 1
        m0 = mb.ConvBNReLU(inp,
                           oup,
                           kernel_size,
                           groups=groups,
                           active_fn=nn.ReLU)
        m0.apply(random_bn)
        mask = torch.tensor([False, True, True, False, False])
        num_remain = mask.sum().item()
        m1 = mb.ConvBNReLU(inp,
                           num_remain,
                           kernel_size,
                           groups=groups,
                           active_fn=nn.ReLU)
        m1.apply(random_bn)
        infos = cu.compress_conv_bn_relu(m1,
                                         m0,
                                         mask,
                                         prefix_new='new',
                                         prefix_old='old')
        inputs = torch.randn(2, 3, 10, 10)
        self._apply_info(infos)

        for m in [m0, m1]:
            m.train()
        lhs = m0(inputs)
        rhs = m1(inputs)
        assertAllClose(lhs[:, mask], rhs)

        for m in [m0, m1]:
            m.eval()
        lhs = m0(inputs)
        rhs = m1(inputs)
        assertAllClose(lhs[:, mask], rhs)


if __name__ == "__main__":
    unittest.main()
