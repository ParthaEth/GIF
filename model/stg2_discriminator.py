import torch
from torch import nn
import math
from model.stylegan2_common_layers import ConvLayer, ResBlock, EqualLinear
from my_utils.graph_writer import graph_writer


class Discriminator(nn.Module):
    def __init__(self, size, channel_multiplier=2, num_color_chnls=3, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        convs = [graph_writer.CallWrapper(ConvLayer(num_color_chnls, channels[size], 1))]

        log_size = int(math.log(size, 2))

        in_channel = channels[size]

        for i in range(log_size, 2, -1):
            out_channel = channels[2 ** (i - 1)]

            convs.append(graph_writer.CallWrapper(ResBlock(in_channel, out_channel, blur_kernel)))

            in_channel = out_channel

        self.convs = nn.Sequential(*convs)

        self.stddev_group = 4
        self.stddev_feat = 1

        self.final_conv = graph_writer.CallWrapper(ConvLayer(in_channel + 1, channels[4], 3))
        self.final_linear = graph_writer.CallWrapper(nn.Sequential(
            EqualLinear(channels[4] * 4 * 4, channels[4], activation='fused_lrelu'),
            EqualLinear(channels[4], 1),
        ))

    def forward(self, input, condition=None, step=0, alpha=0):
        if type(input) in (list, tuple):
            input = input[0]

        if condition is not None:
            input = torch.cat((input, condition), axis=1)
        out = self.convs(input)

        batch, channel, height, width = out.shape
        # import ipdb;
        # ipdb.set_trace()
        group = min(batch, self.stddev_group)
        node_trace_name = getattr(out, '_self_node_tracing_name', None)
        stddev = out.view(group, -1, self.stddev_feat, channel // self.stddev_feat, height, width)
        stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
        stddev = stddev.mean([2, 3, 4], keepdims=True).squeeze(2)
        stddev = stddev.repeat(group, 1, height, width)
        out = torch.cat([out, stddev], 1)
        out._self_node_tracing_name = node_trace_name

        out = self.final_conv(out)
        node_trace_name = getattr(out, '_self_node_tracing_name', None)

        out = out.view(batch, -1)
        out._self_node_tracing_name = node_trace_name

        out = self.final_linear(out)

        return out, None
