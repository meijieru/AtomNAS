"""Supernet builder."""
import numbers
from torch import nn

from models.mobilenet_base import _make_divisible
from models.mobilenet_base import ConvBNReLU
from models.mobilenet_base import get_active_fn
from models.mobilenet_base import get_block
from models.mobilenet_base import _get_named_block_list

__all__ = ['MobileNetV2']


def get_block_wrapper(block_str):
    """Wrapper for MobileNetV2 block.

    Use `expand_ratio` instead of manually specified channels number."""

    class InvertedResidual(get_block(block_str)):

        def __init__(self,
                     inp,
                     oup,
                     stride,
                     expand_ratio,
                     kernel_sizes,
                     active_fn=None,
                     batch_norm_kwargs=None):

            def _expand_ratio_to_hiddens(expand_ratio):
                if isinstance(expand_ratio, list):
                    assert len(expand_ratio) == len(kernel_sizes)
                    expand = True
                elif isinstance(expand_ratio, numbers.Number):
                    expand = expand_ratio != 1
                    expand_ratio = [expand_ratio for _ in kernel_sizes]
                else:
                    raise ValueError(
                        'Unknown expand_ratio type: {}'.format(expand_ratio))
                hidden_dims = [int(round(inp * e)) for e in expand_ratio]
                return hidden_dims, expand

            hidden_dims, expand = _expand_ratio_to_hiddens(expand_ratio)
            super(InvertedResidual,
                  self).__init__(inp,
                                 oup,
                                 stride,
                                 hidden_dims,
                                 kernel_sizes,
                                 expand,
                                 active_fn=active_fn,
                                 batch_norm_kwargs=batch_norm_kwargs)
            self.expand_ratio = expand_ratio

    return InvertedResidual


class MobileNetV2(nn.Module):
    """MobileNetV2-like network."""

    def __init__(self,
                 num_classes=1000,
                 input_size=224,
                 input_channel=32,
                 last_channel=1280,
                 width_mult=1.0,
                 inverted_residual_setting=None,
                 dropout_ratio=0.2,
                 batch_norm_momentum=0.1,
                 batch_norm_epsilon=1e-5,
                 active_fn='nn.ReLU6',
                 block='InvertedResidualChannels',
                 round_nearest=8):
        """Build the network.

        Args:
            num_classes (int): Number of classes
            input_size (int): Input resolution.
            input_channel (int): Number of channels for stem convolution.
            last_channel (int): Number of channels for stem convolution.
            width_mult (float): Width multiplier - adjusts number of channels in
                each layer by this amount
            inverted_residual_setting (list): A list of
                [expand ratio, output channel, num repeat,
                stride of first block, A list of kernel sizes].
            dropout_ratio (float): Dropout ratio for linear classifier.
            batch_norm_momentum (float): Momentum for batch normalization.
            batch_norm_epsilon (float): Epsilon for batch normalization.
            active_fn (str): Specify which activation function to use.
            block (str): Specify which MobilenetV2 block implementation to use.
            round_nearest (int): Round the number of channels in each layer to
                be a multiple of this number Set to 1 to turn off rounding.
        """
        super(MobileNetV2, self).__init__()
        batch_norm_kwargs = {
            'momentum': batch_norm_momentum,
            'eps': batch_norm_epsilon
        }

        self.input_size = input_size
        self.input_channel = input_channel
        self.last_channel = last_channel
        self.num_classes = num_classes
        self.width_mult = width_mult
        self.round_nearest = round_nearest
        self.inverted_residual_setting = inverted_residual_setting
        self.active_fn = active_fn
        self.block = block
        self.batch_norm_kwargs = batch_norm_kwargs

        if len(inverted_residual_setting) == 0 or len(
                inverted_residual_setting[0]) != 5:
            raise ValueError(
                "inverted_residual_setting should be non-empty "
                "or a 5-element list, got {}".format(inverted_residual_setting))
        if input_size % 32 != 0:
            raise ValueError('Input size must divide 32')
        active_fn = get_active_fn(active_fn)
        block = get_block_wrapper(block)

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult,
                                        round_nearest)
        last_channel = _make_divisible(last_channel * max(1.0, width_mult),
                                       round_nearest)
        features = [
            ConvBNReLU(3,
                       input_channel,
                       stride=2,
                       batch_norm_kwargs=batch_norm_kwargs,
                       active_fn=active_fn)
        ]
        # building inverted residual blocks
        for t, c, n, s, ks in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(
                    block(input_channel,
                          output_channel,
                          stride,
                          t,
                          ks,
                          active_fn=active_fn,
                          batch_norm_kwargs=batch_norm_kwargs))
                input_channel = output_channel
        # building last several layers
        features.append(
            ConvBNReLU(input_channel,
                       last_channel,
                       kernel_size=1,
                       batch_norm_kwargs=batch_norm_kwargs,
                       active_fn=active_fn))
        avg_pool_size = input_size // 32
        features.append(nn.AvgPool2d(avg_pool_size))
        # make it nn.Sequential
        self.features = nn.Sequential(*features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_ratio),
            nn.Linear(last_channel, num_classes),
        )

    def get_named_block_list(self):
        """Get `{name: module}` dictionary for all inverted residual blocks."""
        return _get_named_block_list(self)

    def forward(self, x):
        x = self.features(x)
        x = x.squeeze(3).squeeze(2)
        x = self.classifier(x)
        return x


Model = MobileNetV2
