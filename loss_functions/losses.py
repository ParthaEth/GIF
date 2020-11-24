import sys
sys.path.append('../')
import torch
from torch.autograd import grad
import numpy as np
from PIL import Image

import constants as cnst
from model.stg2_generator import FlameTextureSpace
from my_utils.visualize_flame_overlay import OverLayViz
from my_utils.flm_dynamic_fit_overlay import camera_dynamic, camera_ringnetpp
import constants
from dataset_loaders import fast_image_reshape


def l2_reg(model):
    l2_reg = 0
    for param in model.parameters():
        l2_reg += torch.norm(param)
    return l2_reg


def get_shuffled_fmae_batch_loss_for_real_batch(scores):
    # Normally we try to maximize teh scores
    batch_size = scores.shape[0]//5
    # this is loss so will be minimized
    signs = torch.ones(scores.shape).cuda()  # first batch is all real
    signs[batch_size:batch_size*2, 1] = -1  # Maximize everything but shape+imge decision
    signs[batch_size*2:batch_size*3, 2] = -1  # Maximize everything but exp+imge decision
    signs[batch_size*3:batch_size*4, 3] = -1  # Maximize everything but pose+imge decision
    signs[batch_size*4:batch_size*5, 4] = -1  # Maximize everything but camera+imge decision

    # return torch.nn.functional.softplus(scores*signs).mean()
    return scores * signs


def get_disentanglement_pen(outputs_decissions, input_flame_params):
    """outputs_decissions:
       # 5 decissions,
       # 0: realistic image,
       # 1: shape and image together looks good
       # 2: exp and iamge looks good together
       # 3: pose and image looks good together
       # 4: camera and pose and image looks good together

       input_factors:
       0: image
       1: 159 dim flame
            flame[0:100]: Shape
            flame[100:150]: Expression
            flame[150:156]: Pose
            flame[156:159]: Camera"""
    shape = constants.INDICES['SHAPE']
    exp = constants.INDICES['EXP']
    pose = constants.INDICES['POSE']
    camera = constants.INDICES['CAM']

    d_img_d_flame = grad(outputs=outputs_decissions[:, 0].sum(), inputs=input_flame_params,
                         create_graph=True)[0].norm(2, dim=1)

    d_shape_d_exp_pose_cam = grad(outputs=outputs_decissions[:, 1].sum(), inputs=input_flame_params,
                                  create_graph=True)[0][:, exp[0]:].norm(2, dim=1)

    idx_except_exp = list(range(*shape)) + list(range(*pose)) + list(range(*camera))
    d_exp_d_shape_pose_cam = grad(outputs=outputs_decissions[:, 2].sum(), inputs=input_flame_params,
                                  create_graph=True)[0][:, idx_except_exp].norm(2, dim=1)

    idx_except_pose = list(range(*shape)) + list(range(*exp))
    d_pose_d_shape_exp = grad(outputs=outputs_decissions[:, 3].sum(), inputs=input_flame_params,
                              create_graph=True)[0][:, idx_except_pose].norm(2, dim=1)

    idx_except_camera = list(range(*shape)) + list(range(*exp))
    d_cam_d_shape_exp = grad(outputs=outputs_decissions[:, 4].sum(), inputs=input_flame_params,
                             create_graph=True)[0][:, idx_except_camera].norm(2, dim=1)

    return 0.5*(d_img_d_flame + d_shape_d_exp_pose_cam + d_exp_d_shape_pose_cam + d_pose_d_shape_exp +
                d_cam_d_shape_exp)
    # return 0.5*d_img_d_flame


def wgan_gp_loss(predictions):
    loss = predictions - 0.001 * (predictions ** 2)
    loss = -loss
    return loss


def grad_penalty_loss(inputs, outs, step):
    grad_penalty = 0
    for inp_idx, inpt in enumerate(inputs):
        # Usual grad pen. No disentanglement forcing. Just to make Adv. learing stable
        grad_real = grad(outputs=outs.sum(), inputs=inpt, create_graph=True)[0]
        if step is not None:
            grad_pen_weight = 1 + step - inp_idx
            grad_pen_weight = 0.05 / (grad_pen_weight * np.log2(1 + grad_pen_weight))
        else:
            grad_pen_weight = 5.0
        grad_penalty += grad_pen_weight * (grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2)

    return grad_penalty


class PathLengthRegularizor:
    def __init__(self):
        self.pl_moving_mean = 0
        self.pl_decay = 0.01

    def path_length_reg(self, generator, step, alpha, input_indices):
        style = torch.randn((input_indices.shape[0], 159), device=input_indices.device)#generator.flame_dim))
        style.requires_grad = True
        fake_images = generator(input=style, noise=None, step=step, alpha=alpha, input_indices=input_indices)[0]

        # Compute |J*y|.
        # Ref : https://github.com/NVlabs/stylegan2/blob/master/training/loss.py
        pl_noise = torch.randn(fake_images.shape, device=input_indices.device) / np.sqrt(np.prod(fake_images.shape)) # Cross check this line
        pl_grads = grad(outputs=torch.sum(fake_images * pl_noise), inputs=style)[0]
        pl_lengths = torch.mean(torch.sqrt(torch.sum(torch.pow(pl_grads, 2), dim=1))) # Waht are these dimensions that we are averaging

        # Track exponential moving average of |J*y|.
        self.pl_moving_mean = self.pl_moving_mean + self.pl_decay * pl_lengths - self.pl_moving_mean

        # Calculate (|J*y|-a)^2.
        pl_penalty = torch.pow(pl_lengths - self.pl_moving_mean, 2)

        return pl_penalty


class InterpolatedTextureLoss:
    def __init__(self, max_images_in_batch):
        texture_data = np.load(cnst.flame_texture_space_dat_file, allow_pickle=True, encoding='latin1').item()
        self.flm_tex_dec = FlameTextureSpace(texture_data, data_un_normalizer=None).cuda()
        self.flame_visualizer = OverLayViz()
        self.face_region_only_mask = np.array(Image.open(cnst.face_region_mask_file))[:, :, 0:1].transpose((2, 0, 1))
        self.face_region_only_mask = (torch.from_numpy(self.face_region_only_mask.astype('float32'))/255.0)[None, ...]
        print('read face region mask')
        flength = 5000
        cam_t = np.array([0., 0., 0])
        self.camera_params = camera_ringnetpp((512, 512), trans=cam_t, focal=flength)
        self.max_num = max_images_in_batch - 1
        # self.max_num = 5
        self.pairs = []
        for i in range(self.max_num):
            for j in range(i+1, self.max_num):
                self.pairs.append((i, j))

        self.pairs = np.array(self.pairs)

    def pairwise_texture_loss(self, tx1, tx2):
        if self.face_region_only_mask.device != tx1.device:
            self.face_region_only_mask = self.face_region_only_mask.to(tx1.device)

        if self.face_region_only_mask.shape[-1] != tx1.shape[-1]:
            face_region_only_mask = fast_image_reshape(self.face_region_only_mask, tx1.shape[1], tx1.shape[2])
        else:
            face_region_only_mask = self.face_region_only_mask

        # import ipdb; ipdb.set_trace()
        return torch.mean(torch.sigmoid(torch.pow(tx1 - tx2, 2)) * face_region_only_mask[0])
        # return torch.mean(torch.pow(tx1 - tx2, 2) * face_region_only_mask[0])

    def tex_sp_intrp_loss(self, flame_batch, generator, step, alpha, max_ids, normal_maps_as_cond,
                          use_posed_constant_input, rendered_flame_as_condition):

        textures, tx_masks, _ = self.get_image_and_textures(alpha, flame_batch, generator, max_ids, normal_maps_as_cond,
                                                            rendered_flame_as_condition, step, use_posed_constant_input)

        # import ipdb; ipdb.set_trace()

        random_pairs_idxs = np.random.choice(len(self.pairs), self.max_num, replace=False)
        random_pairs = self.pairs[random_pairs_idxs]
        loss = 0
        for cur_pair in random_pairs:
            tx_mask_common = tx_masks[cur_pair[1]] * tx_masks[cur_pair[0]]
            loss += self.pairwise_texture_loss(tx1=textures[cur_pair[0]]*tx_mask_common,
                                               tx2=textures[cur_pair[1]]*tx_mask_common)

        return 16*loss / len(random_pairs)

    def get_image_and_textures(self, alpha, flame_batch, generator, max_ids, normal_maps_as_cond,
                               rendered_flame_as_condition, step, use_posed_constant_input):
        batch_size = flame_batch.shape[0]
        flame_batch = flame_batch[:self.max_num, :]  # Just to limit run time
        # import ipdb; ipdb.set_trace()
        if rendered_flame_as_condition or normal_maps_as_cond:
            shape = flame_batch[:, constants.INDICES['SHAPE'][0]:constants.INDICES['SHAPE'][1]]
            exp = flame_batch[:, constants.INDICES['EXP'][0]:constants.INDICES['EXP'][1]]
            pose = flame_batch[:, constants.INDICES['POSE'][0]:constants.INDICES['POSE'][1]]
            if flame_batch.shape[-1] == 159:  # non DECA params
                flame_params = (shape,
                                exp,
                                pose,
                                flame_batch[:,
                                constants.INDICES['TRANS'][0]:constants.INDICES['TRANS'][1]])  # translation
                norma_map_img, _, _, _, rend_flm = self.flame_visualizer.get_rendered_mesh(flame_params=flame_params,
                                                                                           camera_params=self.camera_params)
                rend_flm = rend_flm / 127 - 1.0
                norma_map_img = norma_map_img * 2 - 1
            elif flame_batch.shape[-1] == 236:  # DECA
                cam = flame_batch[:, constants.DECA_IDX['cam'][0]:constants.DECA_IDX['cam'][1]:]

                # Same lightcode for the whole batch
                light_code = flame_batch[0:1, constants.DECA_IDX['lit'][0]:constants.DECA_IDX['lit'][1]:]
                light_code = light_code.repeat(batch_size, 1)
                light_code = light_code.view((batch_size, 9, 3))

                #same texture code for the whole batch
                texture_code = flame_batch[0:1, constants.DECA_IDX['tex'][0]:constants.DECA_IDX['tex'][1]:]
                texture_code = texture_code.repeat(batch_size, 1)
                # import ipdb; ipdb.set_trace()
                norma_map_img, _, _, _, rend_flm = \
                    self.flame_visualizer.get_rendered_mesh(flame_params=(shape, exp, pose, light_code, texture_code),
                                                            camera_params=cam)
                # rend_flm = rend_flm * 2 - 1
                rend_flm = torch.clamp(rend_flm, 0, 1) * 2 - 1
                norma_map_img = torch.clamp(norma_map_img, 0, 1) * 2 - 1
                rend_flm = fast_image_reshape(rend_flm, height_out=256, width_out=256, mode='bilinear')
                norma_map_img = fast_image_reshape(norma_map_img, height_out=256, width_out=256, mode='bilinear')
            else:
                raise ValueError('Flame prameter format not understood')
        # import ipdb; ipdb.set_trace()
        if use_posed_constant_input:
            pose = flame_batch[:, constants.get_idx_list('GLOBAL_ROT')]
        else:
            pose = None
        fixed_identities = torch.ones(flame_batch.shape[0], dtype=torch.long,
                                      device='cuda') * np.random.randint(0, max_ids)
        if rendered_flame_as_condition and normal_maps_as_cond:
            gen_in = torch.cat((rend_flm, norma_map_img), dim=1)
        elif rendered_flame_as_condition:
            gen_in = rend_flm
        elif normal_maps_as_cond:
            gen_in = norma_map_img
        else:
            gen_in = flame_batch
        generated_image = generator(gen_in, pose=pose, step=step, alpha=alpha, input_indices=fixed_identities)[-1]
        textures, tx_masks = self.flm_tex_dec(generated_image, flame_batch)
        return textures, tx_masks, generated_image