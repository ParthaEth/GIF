import sys
import constants as cnst
sys.path.append('../../')
import torch
from dataset_loaders import fast_image_reshape
from model.stg2_generator import StyledGenerator
import os
import constants
from my_utils.visualize_flame_overlay import OverLayViz
from my_utils.flm_dynamic_fit_overlay import camera_ringnetpp, camera_dynamic
from my_utils.generate_gif import generate_from_flame_sequence
from my_utils.generic_utils import save_set_of_images
import glob
from model.stg2_generator import FlameTextureSpace
from my_utils.eye_centering import position_to_given_location
import numpy as np


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
alpha = 1
normalization_file_path = f'{cnst.output_root}FFHQ_dynamicfit_normalization.npz'
normalization_file_path = f'{cnst.output_root}FFHQ_dynamicfit_normalization_hawen_parms.npz'

if 'resolution' in other_params.files:
    resolution = other_params['resolution']

rendered_flame_as_condition = True
rows = 5
cols = 6
b_size = rows*cols
n_frames = 32
step_max = int(np.log2(resolution) - 2)  # starts from 4X4 hence the -2

torch.manual_seed(7)
generator = StyledGenerator(flame_dim=code_size,
                            embedding_vocab_size=69158,
                            rendered_flame_ascondition=rendered_flame_as_condition,
                            inst_norm=use_inst_norm,
                            normal_maps_as_cond=normal_maps_as_cond,
                            core_tensor_res=core_tensor_res,
                            use_styled_conv_stylegan2=True,
                            n_mlp=8).cuda()

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
overlay_visualizer = OverLayViz()
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
flm_3_sigmaparams_dict = load_3sigma_flame(cnst.flm_3_sigmaparams_dir)


light_texture_id_code_source = np.load('./params.npy', allow_pickle=True).item()
ids_to_pick = [1024,  1467,  1552,  1614,  1904,  238,  327,  495,
               1069,  1468,  1554,  1642,  1914,  259,  355,  663,
               127,   1471,  1565,  1683,  1947,  261,  356,
               1427,  1472,  1571,  1891,  2047,  309,  48,]

count = 0
for identity_index in ids_to_pick:

    light_code = light_texture_id_code_source['light_code'][identity_index]
    texture_code = light_texture_id_code_source['texture_code'][identity_index]
    id = light_texture_id_code_source['identity_indices'][identity_index]

    file_names = list(flm_3_sigmaparams_dict.keys())
    flm_params = []
    for i, flm_name in enumerate(file_names):
        flm_val = np.expand_dims(flm_3_sigmaparams_dict[flm_name], axis=0)
        flm_val = torch.from_numpy(flm_val.astype('float32')).cuda()
        flm_params.append(flm_val)

    # Two more FLAME params for texture extremes
    flm_params.append(flm_val*0)
    file_names.append('-3_albedo')
    flm_params.append(flm_val*0)
    file_names.append('+3_albedo')

    # Further two for lighting
    flm_params.append(flm_val*0)
    file_names.append('-3_light')
    flm_params.append(flm_val*0)
    file_names.append('+3_light')
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
    flm_params = position_to_given_location(flame_decoder, flm_params)

    cam = flm_params[:, constants.DECA_IDX['cam'][0]:constants.DECA_IDX['cam'][1]:]
    shape = flm_params[:, constants.INDICES['SHAPE'][0]:constants.INDICES['SHAPE'][1]]
    exp = flm_params[:, constants.INDICES['EXP'][0]:constants.INDICES['EXP'][1]]
    pose = flm_params[:, constants.INDICES['POSE'][0]:constants.INDICES['POSE'][1]]
    # import ipdb; ipdb.set_trace()
    light_code = torch.from_numpy(light_code).cuda()[None, ...].repeat(exp.shape[0]-2, 1, 1)
    # import ipdb; ipdb.set_trace()
    light_code = torch.cat((light_code, mean_minus_3_sigma_light.view(1,9, 3), mean_plu_3_sigma_light.view(1,9, 3)), dim=0)

    texture_code = torch.from_numpy(texture_code).cuda()[None, ...].repeat(exp.shape[0]-4, 1)
    texture_code_neg_3_sigma = texture_code[0:1, :] * 0
    texture_code_neg_3_sigma[0, 0] -= 3
    texture_code_pos_3_sigma = texture_code[0:1, :] * 0
    texture_code_pos_3_sigma[0, 0] += 3
    texture_code = torch.cat((texture_code, texture_code_neg_3_sigma, texture_code_pos_3_sigma, texture_code[:2, :]), dim=0)

    norma_map_img, _, _, _, rend_flm = \
        overlay_visualizer.get_rendered_mesh(flame_params=(shape, exp, pose, light_code, texture_code),
                                                camera_params=cam)
    rend_flm = torch.clamp(rend_flm, 0, 1) * 2 - 1
    norma_map_img = torch.clamp(norma_map_img, 0, 1) * 2 - 1
    rend_flm = fast_image_reshape(rend_flm, height_out=256, width_out=256, mode='bilinear')
    norma_map_img = fast_image_reshape(norma_map_img, height_out=256, width_out=256, mode='bilinear')


    # Only for testing
    # import ipdb; ipdb.set_trace()
    # rend_flm = torch.from_numpy(np.load('../deca_test_4_flame_rendering.npy')).cuda()[:, :3, :, :]
    # norma_map_img = torch.from_numpy(np.load('../deca_test_4_flame_rendering.npy')).cuda()[:, 3:, :, :]
    # flm_rndrds_trn = np.load('../visual_batch_indices_and_flame_renderings.npz')
    # rend_flm = torch.from_numpy(flm_rndrds_trn['condition_parmas'][:4, :3, :, :]).cuda()
    # norma_map_img = torch.from_numpy(flm_rndrds_trn['condition_parmas'][:4, 3:, :, :]).cuda()

    if normal_maps_as_cond and rendered_flame_as_condition:
        # norma_map_img = norma_map_img * 2 - 1
        gen_in = torch.cat((rend_flm, norma_map_img), dim=1)
    elif normal_maps_as_cond:
        gen_in = norma_map_img
    elif rendered_flame_as_condition:
        gen_in = rend_flm

    save_dir_teaser = os.path.join(save_dir_tsr, f'chosen_ids_rand_images/')
    os.makedirs(save_dir_teaser, exist_ok=True)

    # Make rendered flame with white background
    save_set_of_images(path=save_dir_teaser, prefix='mesh_normals_',
                       images=((norma_map_img + 1) / 2).cpu().numpy(),
                       name_list=file_names)

    save_set_of_images(path=save_dir_teaser, prefix='mesh_textured_',
                       images=((rend_flm + 1) / 2).cpu().numpy(),
                       name_list=file_names)
    discard_list = []

    texture_data = np.load(cnst.flame_texture_space_dat_file, allow_pickle=True, encoding='latin1').item()
    flm_tex_dec = FlameTextureSpace(texture_data=texture_data, data_un_normalizer=None).cuda()

    generated_images_uncorrupt = generate_from_flame_sequence(generator, gen_in, pose=None, step=step_max,
                                                              alpha=alpha, noise=None,
                                                              input_indices=fixed_identity_embeddings*int(id))[-1]
    # Steal Textures
    textures, texture_mask = flm_tex_dec(fast_image_reshape(generated_images_uncorrupt.cuda(), height_out=256,
                                                            width_out=256),
                                         flm_params[:generated_images_uncorrupt.shape[0], ...])
    textures *= texture_mask

    save_set_of_images(path=save_dir_teaser, prefix=str(identity_index) + '_',
                       images=((torch.clamp(generated_images_uncorrupt, -1, 1) + 1) / 2).cpu().numpy(),
                       name_list=file_names)
    # import ipdb; ipdb.set_trace()
    save_set_of_images(path=save_dir_teaser, prefix=str(identity_index) + '_texture_',
                       images=((torch.clamp(textures, -1, 1) + 1) / 2).cpu().numpy(),
                       name_list=file_names)

exit(0)