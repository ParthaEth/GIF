import sys
sys.path.append('../')
import constants as cnst
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F

import random

from model.stylegan2_common_layers import StyledConv, get_w_frm_z, ToRGB
from my_utils.graph_writer import graph_writer

from my_utils.photometric_optimization.models import FLAME
from model import mesh_and_3d_helpers
from my_utils.photometric_optimization import gif_helper
from my_utils.photometric_optimization import util


class ConstantInput(nn.Module):
    def __init__(self, channel, size=4, constant_background=False):
        super().__init__()

        self.input = nn.Parameter(torch.randn(1, channel, size, size))

    def forward(self, input):
        batch = input.shape[0]
        out = self.input.repeat(batch, 1, 1, 1)

        return out


class ImgEmbedding(nn.Module):
    def __init__(self, vector_size, vocab_size=70_000):
        super().__init__()
        # self.embd = nn.Embedding(vocab_size, vector_size)
        self.register_buffer('embd_weight', torch.randn((vocab_size, vector_size)))

    def get_embddings(self):
        return self.embd_weight

    def forward(self, input):
        """input: Must be a vector of indices"""
        return self.embd_weight[input]
        # return torch.randn((input.shape[0], 512), device=input.device)

class StyledConvStyleGAN2(nn.Module):
    def __init__(self, in_chnl, out_chnl, ker_sz, blur_kernel, noise_in_dims, one_conv_block=False,
                 apply_sqrt2_fac_in_eq_lin=False):
        super().__init__()
        self.one_conv_block = one_conv_block

        self.st_cv1 = StyledConv(in_chnl, out_chnl, ker_sz, upsample=not self.one_conv_block,
                                 blur_kernel=blur_kernel, noise_in_dims=noise_in_dims,
                                 apply_sqrt2_fac_in_eq_lin=apply_sqrt2_fac_in_eq_lin)
        if not self.one_conv_block:
            self.st_cv2 = StyledConv(out_chnl, out_chnl, ker_sz, upsample=False, blur_kernel=blur_kernel,
                                     noise_in_dims=noise_in_dims,
                                     apply_sqrt2_fac_in_eq_lin=apply_sqrt2_fac_in_eq_lin)

    def forward(self, input, style, noise=None):
        if self.one_conv_block:
            return self.st_cv1(input, style, noise)

        return self.st_cv2(self.st_cv1(input, style, noise), style, noise)


class Generator(nn.Module):
    def __init__(self, code_dim, core_tensor_res=4, channel_multiplier=2, noise_in_dims=None,
                 apply_sqrt2_fac_in_eq_lin=False):
        super().__init__()

        # changes the architecture too much otherwise
        assert(core_tensor_res < 64)
        assert (code_dim == 512)

        # ex_cha_mult = 2  # Set to 1 to restore original StyleGAN code
        ex_cha_mult = 1  # Set to 1 to restore original StyleGAN code

        self.start_step = int(np.log2(core_tensor_res)) - 2

        with graph_writer.ModuleSpace('Generator'):
            self.const_input = ConstantInput(512, size=core_tensor_res)
            blur_kernel = [1, 3, 3, 1]
            self.progression = nn.ModuleList(
                [graph_writer.CallWrapper(StyledConvStyleGAN2(
                    code_dim, 512*ex_cha_mult, 3, blur_kernel=blur_kernel, one_conv_block=True,
                    noise_in_dims=noise_in_dims, apply_sqrt2_fac_in_eq_lin=apply_sqrt2_fac_in_eq_lin)),  # 4X4
                 graph_writer.CallWrapper(StyledConvStyleGAN2(
                     512*ex_cha_mult, 512*ex_cha_mult, 3, blur_kernel=blur_kernel, noise_in_dims=noise_in_dims,
                     apply_sqrt2_fac_in_eq_lin=apply_sqrt2_fac_in_eq_lin)),
                 graph_writer.CallWrapper(StyledConvStyleGAN2(
                     512*ex_cha_mult, 512*ex_cha_mult, 3,  blur_kernel=blur_kernel, noise_in_dims=noise_in_dims,
                     apply_sqrt2_fac_in_eq_lin=apply_sqrt2_fac_in_eq_lin)),
                 graph_writer.CallWrapper(StyledConvStyleGAN2(
                     512*ex_cha_mult, 512*ex_cha_mult, 3, blur_kernel=blur_kernel, noise_in_dims=noise_in_dims,
                     apply_sqrt2_fac_in_eq_lin=apply_sqrt2_fac_in_eq_lin)),
                 graph_writer.CallWrapper(StyledConvStyleGAN2(
                     512*ex_cha_mult, 256*channel_multiplier*ex_cha_mult, 3, blur_kernel=blur_kernel,
                     noise_in_dims=noise_in_dims, apply_sqrt2_fac_in_eq_lin=apply_sqrt2_fac_in_eq_lin)),
                 graph_writer.CallWrapper(StyledConvStyleGAN2(
                     256*ex_cha_mult*channel_multiplier, 128*channel_multiplier*ex_cha_mult, 3, blur_kernel=blur_kernel,
                     noise_in_dims=noise_in_dims, apply_sqrt2_fac_in_eq_lin=apply_sqrt2_fac_in_eq_lin)),
                 graph_writer.CallWrapper(StyledConvStyleGAN2(
                     128*ex_cha_mult*channel_multiplier, 64*channel_multiplier*ex_cha_mult, 3, blur_kernel=blur_kernel,
                     noise_in_dims=noise_in_dims, apply_sqrt2_fac_in_eq_lin=apply_sqrt2_fac_in_eq_lin)),
                 graph_writer.CallWrapper(StyledConvStyleGAN2(
                     64*channel_multiplier*ex_cha_mult, 32*channel_multiplier*ex_cha_mult, 3,  blur_kernel=blur_kernel,
                     noise_in_dims=noise_in_dims, apply_sqrt2_fac_in_eq_lin=apply_sqrt2_fac_in_eq_lin)),
                 graph_writer.CallWrapper(StyledConvStyleGAN2(
                     32*channel_multiplier*ex_cha_mult, 16*channel_multiplier*ex_cha_mult, 3, blur_kernel=blur_kernel,
                     noise_in_dims=noise_in_dims, apply_sqrt2_fac_in_eq_lin=apply_sqrt2_fac_in_eq_lin)),]
            )

            self.to_rgb = nn.ModuleList(
                [
                    graph_writer.CallWrapper(ToRGB(code_dim*ex_cha_mult, code_dim, upsample=False,
                                                   apply_sqrt2_fac_in_eq_lin=apply_sqrt2_fac_in_eq_lin)),
                    graph_writer.CallWrapper(ToRGB(512*ex_cha_mult, code_dim,
                                                   apply_sqrt2_fac_in_eq_lin=apply_sqrt2_fac_in_eq_lin)),
                    graph_writer.CallWrapper(ToRGB(512*ex_cha_mult, code_dim,
                                                   apply_sqrt2_fac_in_eq_lin=apply_sqrt2_fac_in_eq_lin)),
                    graph_writer.CallWrapper(ToRGB(512*ex_cha_mult, code_dim,
                                                   apply_sqrt2_fac_in_eq_lin=apply_sqrt2_fac_in_eq_lin)),
                    graph_writer.CallWrapper(ToRGB(256*ex_cha_mult * channel_multiplier, code_dim,
                                                   apply_sqrt2_fac_in_eq_lin=apply_sqrt2_fac_in_eq_lin)),
                    graph_writer.CallWrapper(ToRGB(128*ex_cha_mult * channel_multiplier, code_dim,
                                                   apply_sqrt2_fac_in_eq_lin=apply_sqrt2_fac_in_eq_lin)),
                    graph_writer.CallWrapper(ToRGB(64*ex_cha_mult * channel_multiplier, code_dim,
                                                   apply_sqrt2_fac_in_eq_lin=apply_sqrt2_fac_in_eq_lin)),
                    graph_writer.CallWrapper(ToRGB(32*ex_cha_mult * channel_multiplier, code_dim,
                                                   apply_sqrt2_fac_in_eq_lin=apply_sqrt2_fac_in_eq_lin)),
                    graph_writer.CallWrapper(ToRGB(16*ex_cha_mult * channel_multiplier, code_dim,
                                                   apply_sqrt2_fac_in_eq_lin=apply_sqrt2_fac_in_eq_lin)),
                ]
            )

            tot_to_rgb_params = 0
            for discrim_params in self.const_input.parameters():
                tot_to_rgb_params += np.prod(discrim_params.shape)
                # print(f'rgb_{np.prod(discrim_params.shape)}')
            print(f'generator const_input n_params: {tot_to_rgb_params}')

            tot_to_rgb_params = 0
            for discrim_params in self.to_rgb.parameters():
                tot_to_rgb_params += np.prod(discrim_params.shape)
                # print(f'rgb_{np.prod(discrim_params.shape)}')
            print(f'generator to_rgb n_params: {tot_to_rgb_params}')

            tot_prog_params = 0
            for discrim_params in self.progression.parameters():
                tot_prog_params += np.prod(discrim_params.shape)
                # print(f'conv_{np.prod(discrim_params.shape)}')
            print(f'generator progression n_params: {tot_prog_params}')

        # self.blur = Blur()

    def forward(self, style, pose, noise, step=0, alpha=-1, input_indices=None, mixing_range=(-1, -1)):
        if pose is None:
            out = torch.zeros((noise[0].shape[0], 3), device=noise[0].device)
        else:
            # pose: b_s X 3, set of vectors representing axis angles
            out = pose

        if len(style) < 2:
            inject_index = [len(self.progression) + 1]
        else:
            inject_index = random.sample(list(range(step)), len(style) - 1)

        crossover = 0
        rgb_out_all_steps = [None]

        # for i, (conv, to_rgb) in enumerate(zip(self.progression, self.to_rgb)):
        for i in range(self.start_step, len(self.progression)):
            # import ipdb;
            # ipdb.set_trace()
            conv = self.progression[i]
            to_rgb = self.to_rgb[i]
            if mixing_range == (-1, -1):
                if crossover < len(inject_index) and i > inject_index[crossover]:
                    crossover = min(crossover + 1, len(style))

                style_step = style[crossover]

            else:
                if mixing_range[0] <= i <= mixing_range[1]:
                    style_step = style[1]
                else:
                    style_step = style[0]

            if i > self.start_step and step > self.start_step:
                out_prev = out

            if i == self.start_step:
                out = self.const_input(out)

            # import ipdb; ipdb.set_trace()
            out = conv(out, style_step, noise[i])

            if i == step:
                out = to_rgb(out, style_step, rgb_out_all_steps[-1])
                rgb_out_all_steps.append(out)
                break
            else:
                # import ipdb; ipdb.set_trace()
                rgb_out_all_steps.append(to_rgb(out, style_step, rgb_out_all_steps[-1]))

        return rgb_out_all_steps[-1:]  # Keeps the type a list so saving code and everything don't need changing


class StyledGenerator(nn.Module):
    def __init__(self, n_mlp=8, embedding_vocab_size=1,
                 rendered_flame_ascondition=False, normal_maps_as_cond=False,
                 core_tensor_res=4, w_truncation_factor=1.0, apply_sqrt2_fac_in_eq_lin=False):
        super().__init__()

        noise_in_dims = int(rendered_flame_ascondition * 3 + normal_maps_as_cond * 3)
        with graph_writer.ModuleSpace('StyledGenerator'):
            self.core_tensor_res = core_tensor_res
            self.rendered_flame_ascondition = rendered_flame_ascondition
            self.normal_maps_as_cond = normal_maps_as_cond
            self.w_truncation_factor = w_truncation_factor
            self.mean_w = None
            code_dim = 512

            self.generator = graph_writer.CallWrapper(Generator(
                code_dim, core_tensor_res=core_tensor_res, noise_in_dims=noise_in_dims,
                apply_sqrt2_fac_in_eq_lin=apply_sqrt2_fac_in_eq_lin))

            if embedding_vocab_size > 1:
                self.embedding_vocab_size = embedding_vocab_size
                self.image_embedding = ImgEmbedding(vector_size=code_dim,
                                                    vocab_size=self.embedding_vocab_size)
                self.img_embdng = graph_writer.CallWrapper(self.image_embedding)

            style_lin_layers = get_w_frm_z(n_mlp, style_dim=code_dim, lr_mlp=0.01, scale_weight=1.0)

            self.z_to_w = graph_writer.CallWrapper(style_lin_layers, node_tracing_name='Style_Transfom')

            transform_params = 0
            for discrim_params in self.z_to_w.parameters():
                transform_params += np.prod(discrim_params.shape)
            print(f'generator z_to_w n_params: {transform_params}')

    def get_embddings(self):
        return self.image_embedding.get_embddings()

    def forward(
        self,
        input,
        pose=None,
        noise=None,
        step=9,
        alpha=1,
        mean_style=None,
        style_weight=0,
        input_indices=None,
        mixing_range=(-1, -1), ):

        assert(step > np.log2(self.core_tensor_res) - 2)

        styles = []
        if type(input) not in (list, tuple):
            input = [input]

        if self.rendered_flame_ascondition or self.normal_maps_as_cond:
            # styles.append(self.style(self.img_embdng(input_indices)))
            if input_indices is None:
                input_indices = torch.zeros(input[0].shape[0], dtype=torch.long, device=input[0].device)

            if input_indices.dtype == torch.float32:  # if the user wishes to feed zs directly
                styles.append(self.z_to_w(input_indices))
            else:
                w = self.z_to_w(self.img_embdng(input_indices))
                # print(torch.std(w.view((-1,))))
                # import ipdb; ipdb.set_trace()
                if np.abs(self.w_truncation_factor - 1.0) > 0.01:  # significant w truncation
                    if self.mean_w is None:
                        self.mean_w = torch.mean(self.z_to_w(self.get_embddings()), dim=0)
                    styles.append(w + (self.mean_w - w) * (1.0 - self.w_truncation_factor))
                else:
                    styles.append(w)
        else:
            for inp in input:
                inp_modified = inp

                if self.embedding_vocab_size > 1:
                    # styles.append(self.style(torch.cat([inp_modified, self.img_embdng(input_indices)], dim=1)))
                    if input_indices.dtype == torch.float32:
                        styles.append(torch.cat([inp_modified, input_indices], dim=1))
                    else:
                        styles.append(torch.cat([inp_modified, self.img_embdng(input_indices)], dim=1))
                else:
                    styles.append(inp_modified)

        batch = input[0].shape[0]

        # import ipdb; ipdb.set_trace()

        if noise is None:
            noise = []

            for i in range(step + 1):
                size = 4 * 2 ** i
                # noise.append(torch.randn(batch, 3, size, size, device=input[0].device)*0)
                noise.append(torch.zeros(batch, 3, size, size, device=input[0].device))

        if self.rendered_flame_ascondition or self.normal_maps_as_cond:
            for i in range(step + 1):
                # Replace noise at each layer with condition
                size = 4 * 2 ** i
                noise[i] = F.interpolate(input[0], size=(size, size), mode='bilinear', align_corners=False)
                noise[i].input_name = f'cnd_{size}X{size}'

        if mean_style is not None:
            styles_norm = []

            for style in styles:
                styles_norm.append(mean_style + style_weight * (style - mean_style))

            styles = styles_norm

        # import ipdb; ipdb.set_trace()
        # for stl in styles:
        #     print(stl.shape)
        # print('len styles = ' + str(len(styles)))
        return self.generator(styles, pose, noise, step, alpha, input_indices=input_indices, mixing_range=mixing_range)

    def mean_style(self, input):
        style = self.z_to_w(input).user_ratings(0, keepdim=True)

        return style


class FlameTextureSpace(nn.Module):
    def __init__(self, texture_data, data_un_normalizer):
        ''':param texture_data:    pre-computed FLAME texture data'''
        super().__init__()
        self.texture_data = texture_data
        self.data_un_normalizer = data_un_normalizer

        config_obj = util.dict2obj(cnst.flame_config)
        render_utils = gif_helper.render_utils(config_obj)
        self.faces = render_utils.get_flame_faces()
        self.flame = FLAME.FLAME(config_obj)

        self.x_coords = self.texture_data.get('x_coords').astype('int')
        self.y_coords = self.texture_data.get('y_coords').astype('int')
        self.valid_pixel_ids = self.texture_data.get('valid_pixel_ids').astype('int')
        self.valid_pixel_3d_faces = torch.from_numpy(self.texture_data.get('valid_pixel_3d_faces').astype('int')).cuda()
        self.valid_pixel_b_coords = \
            torch.from_numpy(self.texture_data.get('valid_pixel_b_coords').astype('float32')).cuda()

    def forward(self, source_img, flame_params_full):
        if self.data_un_normalizer is not None:
            flame_params_full = self.data_un_normalizer(flame_params_full)

        flame_params = (flame_params_full[:, 0:100],  # shape
                        flame_params_full[:, 100:150],  # exp
                        flame_params_full[:, 150:156],  # pose
                        flame_params_full[:, 156:159])  # camera (scale, x_shift, y_shift)

        shape, expression, pose, camera_params = flame_params
        verts, _, _ = self.flame(shape_params=shape, expression_params=expression, pose_params=pose)
        target_mesh_vertices = verts

        # Compute vertex normals
        trans_verts = mesh_and_3d_helpers.batch_orth_proj(target_mesh_vertices, camera_params)
        trans_verts[:, :, 1:] = -trans_verts[:, :, 1:]
        vertex_normals = mesh_and_3d_helpers.vertex_normals(
            trans_verts, self.faces.expand(shape.shape[0], -1, -1))

        return self.compute_texture_map(source_img, target_mesh_vertices, vertex_normals,
                                        camera_params=camera_params)

    # def compute_texture_map(self, source_img, target_mesh, camera_params):
    def compute_texture_map(self, source_img, target_mesh_v, vertex_normals, camera_params):
        '''
        Given an image and a mesh aligned with the image (under scale-orthographic projection), project the image onto the
        mesh and return a texture map.
        :param source_img:      source image
        :param target_mesh:     mesh in FLAME mesh topology aligned with the source image
        :param camera_params:    scale of mesh for the projection
        :return:                computed texture map
        '''

        # target_mesh_v = target_mesh_v.cpu().numpy()[0]

        pixel_3d_points = \
            target_mesh_v[:, self.valid_pixel_3d_faces[:, 0], :] * self.valid_pixel_b_coords[:, 0][None, :, None] + \
            target_mesh_v[:, self.valid_pixel_3d_faces[:, 1], :] * self.valid_pixel_b_coords[:, 1][None, :, None] + \
            target_mesh_v[:, self.valid_pixel_3d_faces[:, 2], :] * self.valid_pixel_b_coords[:, 2][None, :, None]

        # Uncomment the following line to check numpy texture stealing
        # return self.debug_np_code(camera_params, target_mesh_v, vertex_normals, source_img)

        # proj_2d_points_normalized had pixels between +- 1 since grid requires that. if you unnormalize them yo will
        # get complimentoryask-like image. i.e. don't do
        # proj_2d_points_normalized = * source_img_size/2 + source_img_size/2 * source_img_size/2 +  source_img_size/2
        proj_2d_points_normalized = mesh_and_3d_helpers.batch_orth_proj(pixel_3d_points, camera_params)[:, :, :2]
        proj_2d_points_normalized[:, :, 1] *= -1

        texture_grid = torch.zeros((source_img.shape[0], 256, 256, 2), dtype=torch.float32, device=source_img.device)
        texture_grid[:, self.y_coords[self.valid_pixel_ids], self.x_coords[self.valid_pixel_ids], :] = \
            proj_2d_points_normalized

        texture_img = F.grid_sample(source_img, texture_grid)
        # import ipdb; ipdb.set_trace()

        pixel_3d_normals = \
            vertex_normals[:, self.valid_pixel_3d_faces[:, 0], :] * self.valid_pixel_b_coords[:, 0][None:, None] +\
            vertex_normals[:, self.valid_pixel_3d_faces[:, 1], :] * self.valid_pixel_b_coords[:, 1][None:, None] +\
            vertex_normals[:, self.valid_pixel_3d_faces[:, 2], :] * self.valid_pixel_b_coords[:, 2][None:, None]
        texture_vis_mask = torch.zeros((source_img.shape[0], 1, 256, 256), dtype=torch.bool,
                                       device=source_img.device)
        texture_vis_mask[:, :, self.y_coords[self.valid_pixel_ids], self.x_coords[self.valid_pixel_ids]] = \
            (pixel_3d_normals[:, :, -1:] < 0).transpose(1, 2)

        # return fast_image_reshape(texture_img, source_img.shape[2], source_img.shape[3])
        return texture_img, texture_vis_mask
