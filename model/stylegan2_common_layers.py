import sys
sys.path.append('../')
import math
import random
# import functools
# import operator

import torch
from torch import nn
from torch.nn import functional as F
from my_utils.graph_writer import graph_writer
# from torch.autograd import Function

# from my_utils.op import FusedLeakyReLU
# from my_utils.op import FusedLeakyReLU, fused_leaky_relu, upfirdn2d
# from my_utils.op import upfirdn2d
# import scipy.ndimage.filters as fi
# from math import sqrt
import numpy as np


class FusedLeakyReLU(nn.Module):
    def __init__(self, channel, negative_slope=0.2, scale=2 ** 0.5):
        super().__init__()

        self.bias = nn.Parameter(torch.zeros(1, channel, 1, 1))
        self.negative_slope = negative_slope
        self.scale = scale

        self.activation = nn.LeakyReLU(negative_slope=negative_slope, inplace=False)

    def forward(self, input):
        # x = self.bias.repeat(input.size()[0],1,1,1)
        # import ipdb ; ipdb.set_trace()
        x = self.bias
        biased_input = torch.add(input, x)
        activation_output = self.activation(biased_input)
        scaled_output = self.scale * activation_output
        return scaled_output


def upfirdn2d(input, kernel, up=1, down=1, pad=(0, 0)):
    #     out = UpFirDn2d.apply(
    #         input, kernel, (up, up), (down, down), (pad[0], pad[1], pad[0], pad[1])
    #     )
    _, minor_dim, in_h, in_w = input.size()
    # import ipdb; ipdb.set_trace()
    kernel_h, kernel_w = kernel.size()
    x = input

    up_x, up_y = up, up
    down_x, down_y = down, down
    pad_x0, pad_x1, pad_y0, pad_y1 = pad[0], pad[1], pad[0], pad[1]
    # import ipdb; ipdb.set_trace()
    x = x.view((-1, minor_dim, in_h, 1, in_w, 1))
    pad = (0, up_x - 1, 0, 0, 0, up_y - 1, 0, 0, 0, 0, 0, 0)
    x = F.pad(x, pad, "constant", 0)  # effectively zero padding
    x = x.view((-1, minor_dim, in_h * up_y, in_w * up_x))
    x = F.pad(x, (max(pad_x0, 0), max(pad_x1, 0), max(pad_y0, 0), max(pad_y1, 0), 0, 0, 0, 0), "constant", 0)
    x = x[:, :, max(-pad_y0, 0): x.size()[2] - max(-pad_y1, 0), max(-pad_x0, 0): x.size()[3] - max(-pad_x1, 0)]

    x = x.reshape([-1, 1, in_h * up_y + pad_y0 + pad_y1, in_w * up_x + pad_x0 + pad_x1])

    kernel = torch.flip(kernel, [0, 1]).view(1, 1, kernel_h, kernel_w)
    x = F.conv2d(x, kernel)
    x = x.reshape(
        -1,
        minor_dim,
        in_h * up_y + pad_y0 + pad_y1 - kernel_h + 1,
        in_w * up_x + pad_x0 + pad_x1 - kernel_w + 1,
    )
    return x[:, :, ::down_y, ::down_x]


class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input * torch.rsqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)


def make_kernel(k):
    k = torch.tensor(k, dtype=torch.float32)

    if k.ndim == 1:
        k = k[None, :] * k[:, None]

    k /= k.sum()

    return k


class Upsample(nn.Module):
    def __init__(self, kernel, factor=2):
        super().__init__()
        # import ipdb; ipdb.set_trace()
        self.factor = factor
        kernel = make_kernel(kernel) * (factor ** 2)
        self.register_buffer('kernel', kernel)

        p = kernel.shape[0] - factor

        pad0 = (p + 1) // 2 + factor - 1
        pad1 = p // 2

        self.pad = (pad0, pad1)

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, up=self.factor, down=1, pad=self.pad)

        return out


class Downsample(nn.Module):
    def __init__(self, kernel, factor=2):
        super().__init__()
        # import ipdb; ipdb.set_trace()
        self.factor = factor
        kernel = make_kernel(kernel)
        self.register_buffer('kernel', kernel)

        p = kernel.shape[0] - factor

        pad0 = (p + 1) // 2
        pad1 = p // 2

        self.pad = (pad0, pad1)

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, up=1, down=self.factor, pad=self.pad)

        return out


class Blur(nn.Module):
    def __init__(self, kernel, pad, upsample_factor=1):
        super().__init__()

        kernel = make_kernel(kernel)

        if upsample_factor > 1:
            kernel = kernel * (upsample_factor ** 2)

        self.register_buffer('kernel', kernel)

        self.pad = pad

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, pad=self.pad)

        return out


class EqualConv2d(nn.Module):
    def __init__(
            self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=True
    ):
        super().__init__()

        self.weight = nn.Parameter(
            torch.randn(out_channel, in_channel, kernel_size, kernel_size)
        )
        self.scale = 1 / math.sqrt(in_channel * kernel_size ** 2)

        self.stride = stride
        self.padding = padding

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channel))

        else:
            self.bias = None

    def forward(self, input):
        out = F.conv2d(
            input,
            self.weight * self.scale,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
        )

        return out

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]},'
            f' {self.weight.shape[2]}, stride={self.stride}, padding={self.padding})'
        )


class EqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1, activation=None, scale_weight=1.0,
                 apply_sqrt2_fac_in_eq_lin=False):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul/scale_weight))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))

        else:
            self.bias = None

        self.activation = activation

        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul
        self.apply_sqrt2_fac_in_eq_lin = apply_sqrt2_fac_in_eq_lin

    def forward(self, input):
        if self.activation:
            out = F.linear(input, self.weight * self.scale)
            # out = fused_leaky_relu(out, self.bias * self.lr_mul)
            out = F.leaky_relu(out + self.bias * self.lr_mul, negative_slope=0.2)

            # The following is a accidental overhead. Stg2 multiplies with sqrt 2 while stg1 did not.
            # We ended up training one model with the multiplication and the others without. Due to time reason we
            # could not make it uniform. However this doesn't effect the image quality in any way to our experience.
            if self.apply_sqrt2_fac_in_eq_lin:
                out *= 1.41421356237

        else:
            # import ipdb; ipdb.set_trace()
            out = F.linear(
                input, self.weight * self.scale, bias=self.bias * self.lr_mul
            )

        return out

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})'
        )


class ScaledLeakyReLU(nn.Module):
    def __init__(self, negative_slope=0.2):
        super().__init__()

        self.negative_slope = negative_slope

    def forward(self, input):
        out = F.leaky_relu(input, negative_slope=self.negative_slope)

        return out * math.sqrt(2)


class ModulatedConv2d(nn.Module):
    def __init__(
            self,
            in_channel,
            out_channel,
            kernel_size,
            style_dim,
            demodulate=True,
            upsample=False,
            downsample=False,
            blur_kernel=[1, 3, 3, 1],
            apply_sqrt2_fac_in_eq_lin=False
    ):
        super().__init__()

        self.eps = 1e-8
        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.upsample = upsample
        self.downsample = downsample
        # import ipdb; ipdb.set_trace()
        if upsample:
            factor = 2
            p = (len(blur_kernel) - factor) - (kernel_size - 1)
            pad0 = (p + 1) // 2 + factor - 1
            pad1 = p // 2 + 1

            self.blur = Blur(blur_kernel, pad=(pad0, pad1), upsample_factor=factor)

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            self.blur = Blur(blur_kernel, pad=(pad0, pad1))

        fan_in = in_channel * kernel_size ** 2
        self.scale = 1 / math.sqrt(fan_in)
        self.padding = kernel_size // 2

        self.weight = nn.Parameter(
            torch.randn(1, out_channel, in_channel, kernel_size, kernel_size)
        )

        self.modulation = EqualLinear(style_dim, in_channel, bias_init=1,
                                      apply_sqrt2_fac_in_eq_lin=apply_sqrt2_fac_in_eq_lin)

        self.demodulate = demodulate

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.in_channel}, {self.out_channel}, {self.kernel_size}, '
            f'upsample={self.upsample}, downsample={self.downsample})'
        )

    def forward(self, input, style):
        # import ipdb; ipdb.set_trace()
        batch, in_channel, height, width = input.shape

        style = self.modulation(style).view(batch, 1, in_channel, 1, 1)
        weight = self.scale * self.weight * style

        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
            weight = weight * demod.view(batch, self.out_channel, 1, 1, 1)

        weight = weight.view(
            batch * self.out_channel, in_channel, self.kernel_size, self.kernel_size
        )

        if self.upsample:
            input = input.view(1, batch * in_channel, height, width)
            weight = weight.view(
                batch, self.out_channel, in_channel, self.kernel_size, self.kernel_size
            )
            weight = weight.transpose(1, 2).reshape(
                batch * in_channel, self.out_channel, self.kernel_size, self.kernel_size
            )
            out = F.conv_transpose2d(input, weight, padding=0, stride=2, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)
            out = self.blur(out)

        elif self.downsample:
            input = self.blur(input)
            _, _, height, width = input.shape
            input = input.view(1, batch * in_channel, height, width)
            out = F.conv2d(input, weight, padding=0, stride=2, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        else:
            input = input.view(1, batch * in_channel, height, width)
            out = F.conv2d(input, weight, padding=self.padding, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        return out


# class NoiseInjection(nn.Module):
#     def __init__(self, channel, noise_channels=1):
#         super().__init__()
#
#         self.weight = nn.Parameter(torch.zeros(1))
#         print('Warning: Wrong noide injetion layer. Condition not considered.')
#
#     def forward(self, image, noise=None):
#         # if noise is None:
#         batch, _, height, width = image.shape
#         noise = image.new_empty(batch, 1, height, width).normal_()
#
#         return image + self.weight * noise

# class NoiseInjection(nn.Module):
#     '''Just for run_id 9'''
#     def __init__(self, noise_in_chalnnels, noise_out_channels):
#         super().__init__()
#
#         self.groups = 1
#         kernel_size = (3, 3)
#         # self.weight = nn.Parameter(torch.FloatTensor(channel, noise_channels // self.groups, *kernel_size).uniform_(-1, 1))
#         # self.weight = nn.Parameter(torch.randn(channel, noise_channels // self.groups, *kernel_size))
#         # self.register_buffer('weight', torch.randn((channel, noise_channels // self.groups, *kernel_size)))
#         self.register_parameter('weight',
#                                 torch.nn.Parameter(torch.zeros((noise_out_channels, 1 // self.groups, *kernel_size))))
#         # self.bias = nn.Parameter(torch.Tensor(channel))
#         # self.weight = nn.Conv2d(noise_channels, channel, 3)
#
#     def forward(self, image, noise):
#         # import ipdb; ipdb.set_trace()
#         # return image + self.weight * self.noise_conv(noise)
#         convolved_noise = F.conv2d(noise, self.weight.repeat(1, 3, 1, 1),
#                                    bias=None, stride=1, padding=1, dilation=1, groups=self.groups)
#         return image + convolved_noise

class NoiseInjection(nn.Module):
    @staticmethod
    def small_init_weights(m):
        if hasattr(m, 'weight'):
            m.weight.data = torch.randn_like(m.weight)/100
        if hasattr(m, 'bias'):
            m.bias.data.fill_(0.0001)

    def __init__(self, noise_in_chalnnels, noise_out_channels):
        super().__init__()

        # self.groups = 1
        # kernel_size = (3, 3)
        # self.register_parameter('weight',
        #                         nn.Parameter(torch.zeros((noise_out_channels // self.groups, 1, *kernel_size))))

        self.noise_in_chalnnels = noise_in_chalnnels
        self.noise_conv = nn.Sequential(
            nn.Conv2d(in_channels=noise_in_chalnnels, out_channels=2*noise_in_chalnnels, kernel_size=3, padding=1,
                      dilation=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=2*noise_in_chalnnels, out_channels=4 * noise_in_chalnnels, kernel_size=3, padding=1,
                      dilation=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=4*noise_in_chalnnels, out_channels=noise_out_channels, kernel_size=3, padding=1,
                      dilation=1),
        )
        # self.noise_conv = nn.Sequential(
        #     nn.Conv2d(in_channels=noise_in_chalnnels, out_channels=noise_out_channels, kernel_size=3, padding=1,
        #               dilation=1),
        # )
        self.noise_conv.apply(NoiseInjection.small_init_weights)

    def forward(self, image, noise):
        batch, _, height, width = image.shape
        # import ipdb; ipdb.set_trace()
        if noise is None:
            noise = image.new_empty(batch, self.noise_in_chalnnels, height, width).normal_()

        # convolved_noise = F.conv2d(noise, self.weight.repeat((1, noise.shape[1], 1, 1)), bias=None, stride=1, padding=1,
        #                            dilation=1, groups=self.groups)

        convolved_noise = self.noise_conv(noise)
        return image + convolved_noise


class ConstantInput(nn.Module):
    def __init__(self, channel, size=4):
        super().__init__()

        self.input = nn.Parameter(torch.randn(1, channel, size, size))

    def forward(self, input):
        batch = input.shape[0]
        out = self.input.repeat(batch, 1, 1, 1)

        return out


class StyledConv(nn.Module):
    def __init__(
            self,
            in_channel,
            out_channel,
            kernel_size,
            noise_in_dims,
            style_dim=512,
            upsample=False,
            blur_kernel=[1, 3, 3, 1],
            demodulate=True,
            apply_sqrt2_fac_in_eq_lin=False
    ):
        super().__init__()

        self.conv = ModulatedConv2d(
            in_channel,
            out_channel,
            kernel_size,
            style_dim,
            upsample=upsample,
            blur_kernel=blur_kernel,
            demodulate=demodulate,
            apply_sqrt2_fac_in_eq_lin=apply_sqrt2_fac_in_eq_lin
        )

        self.noise = NoiseInjection(noise_in_dims, out_channel)
        # self.bias = nn.Parameter(torch.zeros(1, out_channel, 1, 1))
        # self.activate = ScaledLeakyReLU(0.2)
        self.activate = FusedLeakyReLU(out_channel)
        # self.activate = nn.LeakyReLU()

    def forward(self, input, style, noise=None):
        # import ipdb; ipdb.set_trace()
        out = self.conv(input, style)
        out = self.noise(out, noise=noise)
        # out = out + self.bias
        out = self.activate(out)

        return out


class ToRGB(nn.Module):
    def __init__(self, in_channel, style_dim, upsample=True, blur_kernel=[1, 3, 3, 1],
                 apply_sqrt2_fac_in_eq_lin=False):
        super().__init__()

        if upsample:
            # self.upsample = Upsample(blur_kernel)
            self.upsample = Upsample(blur_kernel)

        self.conv = ModulatedConv2d(in_channel, 3, 1, style_dim, demodulate=False,
                                    apply_sqrt2_fac_in_eq_lin=apply_sqrt2_fac_in_eq_lin)
        self.bias = nn.Parameter(torch.zeros(1, 3, 1, 1))

    def forward(self, input, style, skip=None):
        out = self.conv(input, style)
        out = out + self.bias

        if skip is not None:
            skip = self.upsample(skip)

            out = out + skip

        return out


def get_w_frm_z(n_mlp, style_dim, lr_mlp=1, scale_weight=1.0):
    if n_mlp > 0:
        layers = [PixelNorm()]

        for i in range(n_mlp):
            layers.append(
                EqualLinear(style_dim, style_dim, lr_mul=lr_mlp, activation='fused_lrelu', scale_weight=scale_weight)
            )

        style = nn.Sequential(*layers)
        return style
    else:
        class Net(nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, *args):
                return args[0]

        return Net()


class Generator(nn.Module):
    def __init__(
            self,
            size,
            style_dim,
            n_mlp,
            channel_multiplier=2,
            blur_kernel=[1, 3, 3, 1],
            lr_mlp=0.01,
    ):
        super().__init__()

        self.size = size

        self.style_dim = style_dim

        with graph_writer.ModuleSpace('Generator'):

            self.style = get_w_frm_z(n_mlp, style_dim, lr_mlp)

            self.channels = {
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

            self.input = graph_writer.CallWrapper(ConstantInput(self.channels[4]))
            self.conv1 = graph_writer.CallWrapper(StyledConv(
                self.channels[4], self.channels[4], 3, style_dim, blur_kernel=blur_kernel, noise_in_dims=1,
            ))
            self.to_rgb1 = graph_writer.CallWrapper(ToRGB(self.channels[4], style_dim, upsample=False))

            self.log_size = int(math.log(size, 2))
            self.num_layers = (self.log_size - 2) * 2 + 1

            self.convs = nn.ModuleList()
            self.upsamples = nn.ModuleList()
            self.to_rgbs = nn.ModuleList()
            self.noises = nn.Module()

            in_channel = self.channels[4]

            for layer_idx in range(self.num_layers):
                res = (layer_idx + 5) // 2
                shape = [1, 1, 2 ** res, 2 ** res]
                self.noises.register_buffer(f'noise_{layer_idx}', torch.randn(*shape))

            for i in range(3, self.log_size + 1):
                out_channel = self.channels[2 ** i]

                self.convs.append(
                    graph_writer.CallWrapper(StyledConv(in_channel, out_channel, 3, style_dim, upsample=True,
                                             blur_kernel=blur_kernel, noise_in_dims=1,))
                )

                self.convs.append(
                    graph_writer.CallWrapper(StyledConv(
                        out_channel, out_channel, 3, style_dim, blur_kernel=blur_kernel, noise_in_dims=1,
                    ))
                )

                self.to_rgbs.append(graph_writer.CallWrapper(ToRGB(out_channel, style_dim)))

                in_channel = out_channel

            self.n_latent = self.log_size * 2 - 2

            tot_prog_params = 0
            for discrim_params in self.noises.parameters():
                tot_prog_params += np.prod(discrim_params.shape)
                # print(f'noises_{np.prod(discrim_params.shape)}')
            print(f'generator static noises n_params: {tot_prog_params}')

            tot_prog_params = 0
            for discrim_params in self.upsamples.parameters():
                tot_prog_params += np.prod(discrim_params.shape)
                # print(f'noises_{np.prod(discrim_params.shape)}')
            print(f'generator upsamples n_params: {tot_prog_params}')

            to_rgb_params_cnt = 0
            for to_rgb_params in self.to_rgb1.parameters():
                to_rgb_params_cnt += np.prod(to_rgb_params.shape)
                # print(f'rgb_{np.prod(to_rgb_params.shape)}')
            for to_rgb_params in self.to_rgbs.parameters():
                to_rgb_params_cnt += np.prod(to_rgb_params.shape)
                # print(f'rgb_{np.prod(to_rgb_params.shape)}')
            print(f'generator to_rgb_params n_params: {to_rgb_params_cnt}')

            conv_params_cnt = 0
            for conv_param in self.conv1.parameters():
                conv_params_cnt += np.prod(conv_param.shape)
                # print(f'conv_{np.prod(conv_param.shape)}')
            for conv_param in self.convs.parameters():
                conv_params_cnt += np.prod(conv_param.shape)
                # print(f'conv_{np.prod(conv_param.shape)}')
            print(f'generator conv_params_cnt n_params: {conv_params_cnt}')

    def make_noise(self):
        device = self.input.input.device

        ns1 = torch.randn(1, 1, 2 ** 2, 2 ** 2, device=device)
        ns1.input_name = 'cnd_2X2'
        noises = [ns1]

        for i in range(3, self.log_size + 1):
            for _ in range(2):
                sz = 2 ** i
                ns = torch.randn(1, 1, sz, sz, device=device)
                ns.input_name = f'cnd_{sz}X{sz}'
                noises.append(ns)

        return noises

    def mean_latent(self, n_latent):
        latent_in = torch.randn(
            n_latent, self.style_dim, device=self.input.input.device
        )
        latent = self.style(latent_in).mean(0, keepdim=True)

        return latent

    def get_latent(self, input):
        return self.style(input)

    def forward(
            self,
            styles,
            return_latents=False,
            inject_index=None,
            truncation=1,
            truncation_latent=None,
            input_is_latent=False,
            noise=None,
            randomize_noise=True,
    ):
        if not input_is_latent:
            styles = [self.style(s) for s in styles]

        if noise is None:
            if randomize_noise:
                noise = [None] * self.num_layers
            else:
                noise = []
                for i in range(self.num_layers):
                    ns = getattr(self.noises, f'noise_{i}')
                    _, _, height, width = ns.shape
                    ns.input_name = f'cond_{height}X{width}'
                    noise.append(ns)

        if truncation < 1:
            style_t = []

            for style in styles:
                style_t.append(truncation_latent + truncation * (style - truncation_latent))

            styles = style_t

        if len(styles) < 2:
            inject_index = self.n_latent

            if styles[0].ndim < 3:
                latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)

            else:
                latent = styles[0]

        else:
            if inject_index is None:
                inject_index = random.randint(1, self.n_latent - 1)

            latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
            latent2 = styles[1].unsqueeze(1).repeat(1, self.n_latent - inject_index, 1)

            latent = torch.cat([latent, latent2], 1)

        out = self.input(latent)
        st_0 = latent[:, 0]
        st_0.input_name = 'style'
        out = self.conv1(out, st_0, noise=noise[0])

        skip = self.to_rgb1(out, latent[:, 1])

        i = 1
        for conv1, conv2, noise1, noise2, to_rgb in zip(
                self.convs[::2], self.convs[1::2], noise[1::2], noise[2::2], self.to_rgbs
        ):

            st_i = latent[:, i]
            st_i.input_name = 'style'
            out = conv1(out, st_i, noise=noise1)

            st_i_p_1 = latent[:, i + 1]
            st_i_p_1.input_name = 'style'
            out = conv2(out, st_i_p_1, noise=noise2)

            st_i_p_2 = latent[:, i + 2]
            st_i_p_2.input_name = 'style'
            skip = to_rgb(out, st_i_p_2, skip)

            i += 2

        image = skip

        if return_latents:
            return image, latent

        else:
            return image, None


class ConvLayer(nn.Sequential):
    def __init__(
            self,
            in_channel,
            out_channel,
            kernel_size,
            downsample=False,
            blur_kernel=[1, 3, 3, 1],
            bias=True,
            activate=True,
    ):
        layers = []

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            layers.append(Blur(blur_kernel, pad=(pad0, pad1)))

            stride = 2
            self.padding = 0

        else:
            stride = 1
            self.padding = kernel_size // 2

        layers.append(
            EqualConv2d(
                in_channel,
                out_channel,
                kernel_size,
                padding=self.padding,
                stride=stride,
                bias=bias and not activate,
            )
        )

        if activate:
            if bias:
                layers.append(FusedLeakyReLU(out_channel))
                # layers.append(nn.LeakyReLU())

            else:
                layers.append(ScaledLeakyReLU(0.2))

        super().__init__(*layers)


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        self.conv1 = ConvLayer(in_channel, in_channel, 3)
        self.conv2 = ConvLayer(in_channel, out_channel, 3, downsample=True)

        self.skip = ConvLayer(
            in_channel, out_channel, 1, downsample=True, activate=False, bias=False
        )

    def forward(self, input):
        out = self.conv1(input)
        out = self.conv2(out)

        skip = self.skip(input)
        out = (out + skip) / math.sqrt(2)

        return out
