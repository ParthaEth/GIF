import sys
sys.path.append('../../../')
import argparse
from configurations import update_config
import constants as cnst
import tqdm
import numpy as np
import torch
from dataset_loaders import fast_image_reshape
from model.stg2_generator import StyledGenerator
import os
import constants
from my_utils.visualize_flame_overlay import OverLayViz
from my_utils.flm_dynamic_fit_overlay import camera_ringnetpp, camera_dynamic
from my_utils.generate_gif import save_images_to_gif, generate_from_flame_sequence
from my_utils.generic_utils import save_set_of_images
import glob
# from model import FlameTextureSpace
from my_utils.eye_centering import position_to_given_location


def linear_interpolate(start, stop, n_steps):
    steps = torch.linspace(0, 1, n_steps).to(start.device)[..., None]
    interp_batch = start.repeat(n_steps, 1)*(1 - steps) + steps*stop.repeat(n_steps, 1)
    return interp_batch


def load_3sigma_flame(directory):
    dirs = ['exp', 'pose', 'shape']

    flame_dict = {}
    for child_directory in dirs:
        for file in glob.glob(os.path.join(directory, child_directory) + '/*.npz'):
            file_vals = np.load(file, allow_pickle=True)
            flame_dict[os.path.basename(file.split('.')[0]) + '_' + child_directory] = \
                np.hstack((file_vals['shape_params'], file_vals['exp_params'], file_vals['pose_params'],
                           np.zeros((3,))))

    return flame_dict



ignore_global_rotation = False
code_size = 159
resolution = 256
overlay_flame_landmarks = True
# flame_version = 'FLAME_2020_revisited'
flame_version = 'DECA'
with_neck = False
random_background = False
# texture_pattern = 'CHKR_BRD'
texture_pattern = 'MEAN_TEXTURE_WITH_CHKR_BOARD'
# texture_pattern = 'CHKR_BRD_FLT_TEETH'
use_inst_norm = True
normal_maps_as_cond = True
use_posed_constant_input = False
core_tensor_res = 4
run_id = 29
model_idx = '026000_1'
ckpt = torch.load(f'{cnst.output_root}checkpoint/{run_id}/{model_idx}.model')
other_params = np.load(f'{cnst.output_root}checkpoint/{run_id}/{model_idx}.npz')
alpha = float(other_params['alpha'])

if 'resolution' in other_params.files:
    resolution = other_params['resolution']

rendered_flame_as_condition = True
rows = 5
cols = 6
b_size = rows*cols
n_frames = 32
step_max = int(np.log2(resolution) - 2)  # starts from 4X4 hence the -2

torch.manual_seed(7)

parser = argparse.ArgumentParser(description='Progressive Growing of GANs')
args, dataset, flame_param_est = update_config(parser)

generator = StyledGenerator(embedding_vocab_size=args.embedding_vocab_size,
                            rendered_flame_ascondition=args.rendered_flame_as_condition,
                            normal_maps_as_cond=args.normal_maps_as_cond,
                            core_tensor_res=args.core_tensor_res,
                            n_mlp=args.nmlp_for_z_to_w).cuda()

# embeddings = generator.get_embddings()
generator = torch.nn.DataParallel(generator)
generator.load_state_dict(ckpt['generator_running'])
# generator.load_state_dict(ckpt['generator'])
# generator.eval()

log_dir = os.path.join(cnst.output_root, 'gif_smpls/FFHQ')

if random_background:
    torch.manual_seed(2)
    back_ground_noise = (torch.randn((3, 224, 224), dtype=torch.float32)*255).clamp(min=0, max=255).cuda()
else:
    back_ground_noise = None
# # Don't provide add rnadom noise to background here. Cause then every frame will have different noise and that's bad
overlay_visualizer = OverLayViz(full_neck=with_neck, add_random_noise_to_background=False, inside_mouth_faces=True,
                                background_img=back_ground_noise, texture_pattern_name=texture_pattern,
                                flame_version=flame_version, image_size=256)
# overlay_visualizer.setup_renderer(mesh_file=None)

if flame_version == 'FLAME_2020_revisited':
    flength = 5000
    cam_t = np.array([0., 0., 0])
    camera_params = camera_ringnetpp((512, 512), trans=cam_t, focal=flength)
elif flame_version == 'DECA':
    pass
else:
    cam_t = np.array([0., 0., 2.5])
    camera_params = camera_dynamic((resolution, resolution), cam_t)

fixed_identity_embeddings = torch.ones(1, dtype=torch.long, device='cuda')

expt_name = 'teaser_figure'
save_root = f'{cnst.output_root}sample/{run_id}'
save_dir_tsr = f'{save_root}/{expt_name}'
os.makedirs(save_dir_tsr, exist_ok=True)
num_frames = 64
interplation_pairs = ['-3_00_00_exp', '+3_00_00_exp',
                      '00_-3_00_exp', '00_+3_00_exp',
                      '00_00_-3_exp', '00_00_+3_exp',
                      'comp4_-pi_8_pose', 'comp4_+pi_8_pose',
                      'comp6_0_pose', 'comp6_+pi_12_pose',
                      '00_-3_00_shape', '00_+3_00_shape',
                      '-3_00_00_shape', '+3_00_00_shape',
                      '00_00_-3_shape', '00_00_+3_shape',]
flm_params = []
for i, flm_name in enumerate(interplation_pairs):
    flm_val = np.expand_dims(cnst.flm_3_sigmaparams_dict[flm_name], axis=0)
    flm_val = torch.from_numpy(flm_val.astype('float32')).cuda()

    # flm_val = eye_reg_cnt.regress_translate(flm_val)  #a different regression based centering
    flm_params.append(flm_val)

# Two more FLAME params for texture extremes
flm_params.append(flm_val*0)
interplation_pairs.append('-3_albedo')
flm_params.append(flm_val*0)
interplation_pairs.append('+3_albedo')

# Further two for lighting
flm_params.append(flm_val*0)
interplation_pairs.append('-3_light')
flm_params.append(flm_val*0)
interplation_pairs.append('+3_light')
mean_lighting = np.array([[3.60890770, 4.02641960, 4.75345130,  0.0709928,  0.08925686,  0.09803673,
                           0.11676598, 0.15575520, 0.20316169, -0.2203714, -0.38735074, -0.63142025,
                           0.00963507, 0.02998208, 0.03484832, -0.0461808, -0.05883689, -0.06856259,
                           0.02707223, 0.07585122, 0.05772701,  0.1662246,  0.20778911,  0.24815214,
                           0.22855483, 0.32920238, 0.52000016,]])
highest_variance_cmp_idx = 2
highest_variance = 0.9143507
mean_plu_3_sigma_light = mean_lighting
mean_plu_3_sigma_light[0, highest_variance_cmp_idx] += 2*highest_variance
mean_plu_3_sigma_light = torch.from_numpy(mean_plu_3_sigma_light.astype('float32')).cuda()
# flame_lighting_plus_3_sigma = np.concatenate((np.zeros((1, flm_val.shape[1]-len(mean_lighting))), mean_plu_3_sigma_light),
#                                              axis=1)
mean_minus_3_sigma_light = mean_lighting
mean_minus_3_sigma_light[0, highest_variance_cmp_idx] -= 2*highest_variance
mean_minus_3_sigma_light = torch.from_numpy(mean_minus_3_sigma_light.astype('float32')).cuda()


flm_params = torch.cat(flm_params, dim=0)
flame_decoder = overlay_visualizer.deca.flame.eval()
# flm_params = position_to_given_location(flame_decoder, flm_params)

random_light_texture_code_idx = 6
fl_param_dict = np.load(cnst.all_fklame_params_file, allow_pickle=True).item()
for i, key in enumerate(fl_param_dict):
    flame_param = fl_param_dict[key]
    if i == random_light_texture_code_idx:
        light_code, texture_code = flame_param['lit'].flatten(), flame_param['tex']
        break


# import ipdb; ipdb.set_trace()
light_code = torch.from_numpy(light_code).cuda()[None, ...].repeat(flm_params.shape[0]-2, 1)
# import ipdb; ipdb.set_trace()
light_code = torch.cat((light_code, mean_minus_3_sigma_light, mean_plu_3_sigma_light), dim=0)

texture_code = torch.from_numpy(texture_code).cuda()[None, ...].repeat(flm_params.shape[0]-4, 1)
texture_code_neg_3_sigma = texture_code[0:1, :] * 0
texture_code_neg_3_sigma[0, 0] -= 3
texture_code_pos_3_sigma = texture_code[0:1, :] * 0
texture_code_pos_3_sigma[0, 0] += 3
texture_code = torch.cat((texture_code, texture_code_neg_3_sigma, texture_code_pos_3_sigma, texture_code[:2, :]), dim=0)

for interp_type_idx in tqdm.tqdm(range(0, len(interplation_pairs), 2)):
    interpolation_type = interplation_pairs[interp_type_idx] + '_VS_' + interplation_pairs[interp_type_idx + 1]
    flame_interp_batch = linear_interpolate(flm_params[interp_type_idx], flm_params[interp_type_idx+1], num_frames)
    flame_interp_batch = position_to_given_location(flame_decoder, flame_interp_batch)
    cam = flame_interp_batch[:, constants.DECA_IDX['cam'][0]:constants.DECA_IDX['cam'][1]:]
    shape = flame_interp_batch[:, constants.INDICES['SHAPE'][0]:constants.INDICES['SHAPE'][1]]
    exp = flame_interp_batch[:, constants.INDICES['EXP'][0]:constants.INDICES['EXP'][1]]
    pose = flame_interp_batch[:, constants.INDICES['POSE'][0]:constants.INDICES['POSE'][1]]
    texture_code_batch = linear_interpolate(texture_code[interp_type_idx], texture_code[interp_type_idx + 1], num_frames)
    light_code_batch = linear_interpolate(light_code[interp_type_idx], light_code[interp_type_idx + 1], num_frames)
    light_code_batch = light_code_batch.view(num_frames, 9, 3)

    norma_map_img, _, _, _, rend_flm = \
        overlay_visualizer.get_rendered_mesh(flame_params=(shape, exp, pose, light_code_batch, texture_code_batch),
                                             camera_params=cam)
    rend_flm = torch.clamp(rend_flm, 0, 1) * 2 - 1
    norma_map_img = torch.clamp(norma_map_img, 0, 1) * 2 - 1
    rend_flm = fast_image_reshape(rend_flm, height_out=256, width_out=256, mode='bilinear')
    norma_map_img = fast_image_reshape(norma_map_img, height_out=256, width_out=256, mode='bilinear')

    if normal_maps_as_cond and rendered_flame_as_condition:
        # norma_map_img = norma_map_img * 2 - 1
        gen_in = torch.cat((rend_flm, norma_map_img), dim=1)
    elif normal_maps_as_cond:
        gen_in = norma_map_img
    elif rendered_flame_as_condition:
        gen_in = rend_flm

    # import ipdb; ipdb.set_trace()
    id_start = 20
    save_dir_teaser = os.path.join(save_dir_tsr, 'interpolations', interpolation_type,
                                   f'images_id_{id_start}_tex_{random_light_texture_code_idx}/')
    os.makedirs(save_dir_teaser, exist_ok=True)

    if not interpolation_type.find('light') >= 0 and not interpolation_type.find('albedo') >= 0:
        norma_map_img_to_save, _, _, _, rend_flm_to_save = \
            overlay_visualizer.get_rendered_mesh(flame_params=(shape, exp, pose, light_code_batch, texture_code_batch),
                                                 camera_params=cam, cull_backfaces=True, grey_texture=True)
        rend_flm_to_save = torch.clamp(rend_flm_to_save, 0, 1) * 2 - 1
        norma_map_img_to_save = torch.clamp(norma_map_img_to_save, 0, 1) * 2 - 1
    else:
        norma_map_img_to_save = norma_map_img
        rend_flm_to_save = rend_flm

    # Make rendered flame with white background
    save_set_of_images(path=os.path.join(save_dir_teaser, 'meshes'), prefix='mesh_normals_',
                       images=((norma_map_img_to_save + 1) / 2).cpu().numpy())

    save_set_of_images(path=os.path.join(save_dir_teaser, 'meshes'), prefix='mesh_textured_',
                       images=((rend_flm_to_save + 1) / 2).cpu().numpy())
    count = 0
    # discard_list = [23, 27, 28, 30, 34, 41, 43, 50, 53, 71, 73, 77, 78, 79, 10, 15, 52, 54, 55, 63, 66, 79, 80, 95, 103,
    #                 106, 107, 112, 119, 121, 123, 125, 137, 141, 143, 144, 155, 161, 165, 190, 194, 174, 176, 180, 182]
    discard_list = []

    texture_data = np.load(cnst.flame_texture_space_dat_file, allow_pickle=True, encoding='latin1').item()
    # flm_tex_dec = FlameTextureSpace(texture_data=texture_data, data_un_normalizer=None).cuda()

    for id in tqdm.tqdm(range(id_start, id_start+20)):
        if id in discard_list:
            continue
        generated_images_uncorrupt = generate_from_flame_sequence(generator, gen_in, pose=None, step=step_max,
                                                                  alpha=alpha, noise=None,
                                                                  input_indices=fixed_identity_embeddings*id)[-1]
        # # Steal Textures
        # textures, texture_mask = flm_tex_dec(fast_image_reshape(generated_images_uncorrupt.cuda(), height_out=256,
        #                                                         width_out=256),
        #                                      flm_params[:generated_images_uncorrupt.shape[0], ...])
        # textures *= texture_mask

        save_set_of_images(path=os.path.join(save_dir_teaser, 'images', str(count)), prefix=str(count) + '_',
                           images=((torch.clamp(generated_images_uncorrupt, -1, 1) + 1) / 2).cpu().numpy())
        # import ipdb; ipdb.set_trace()
        # save_set_of_images(path=save_dir_teaser, prefix=str(count) + '_texture_',
        #                    images=((torch.clamp(textures, -1, 1) + 1) / 2).cpu().numpy(),
        #                    name_list=file_names)
        count += 1
        if count == 200:
            break
exit(0)