import constants as cnst
import os
os.environ['PYTHONHASHSEED'] = '2'
import tqdm
from model.stg2_generator import StyledGenerator
import numpy as np
from my_utils.visualize_flame_overlay import OverLayViz
from my_utils.flm_dynamic_fit_overlay import camera_ringnetpp
from my_utils.generic_utils import save_set_of_images
from my_utils import compute_fid
import constants
from dataset_loaders import fast_image_reshape
import torch
from my_utils import generic_utils
from my_utils.eye_centering import position_to_given_location
from copy import deepcopy
from my_utils.photometric_optimization.models import FLAME
from my_utils.photometric_optimization import util


def ge_gen_in(flm_params, textured_rndr, norm_map, normal_map_cond, texture_cond):
    if normal_map_cond and texture_cond:
        return torch.cat((textured_rndr, norm_map), dim=1)
    elif normal_map_cond:
        return norm_map
    elif texture_cond:
        return textured_rndr
    else:
        return flm_params


def corrupt_flame_given_sigma(flm_params, corruption_type, sigma, jaw_sigma, pose_sigma):
    # import ipdb; ipdb.set_trace()
    # np.random.seed(2)
    corrupted_flame = deepcopy(flm_params)
    if corruption_type == 'shape' or corruption_type == 'all':
        corrupted_flame[:, :10] = flm_params[:, :10] + \
                                   np.clip(np.random.normal(0, sigma, flm_params[:, :10].shape),
                                           -3 * sigma, 3 * sigma).astype('float32')
    if corruption_type == 'exp_jaw'or corruption_type == 'all':
        # Expression
        corrupted_flame[:, 100:110] = flm_params[:, 100:110] + \
                                      np.clip(np.random.normal(0, sigma, flm_params[:, 100:110].shape),
                                              -3 * sigma, 3 * sigma).astype('float32')
        # Jaw pose
        corrupted_flame[:, 153] = flm_params[:, 153] + \
                                  np.random.normal(0, jaw_sigma, corrupted_flame.shape[0])

    if corruption_type == 'pose' or corruption_type == 'all':
        # pose_perturbation = np.random.normal(0, pose_sigma[i], (corrupted_flame.shape[0], 3))
        # corrupted_flame[:, 150:153] += np.clip(pose_perturbation, -3 * pose_sigma[i], 3 * pose_sigma[i])
        pose_perturbation = np.random.normal(0, pose_sigma, (corrupted_flame.shape[0],))
        corrupted_flame[:, 151] = flm_params[:, 151] + \
                                   np.clip(pose_perturbation, -3 * pose_sigma, 3 * pose_sigma)

    return corrupted_flame


# General settings
save_images = True
code_size = 236
use_inst_norm = True
core_tensor_res = 4
resolution = 256
alpha = 1
step_max = int(np.log2(resolution) - 2)
root_out_dir = f'{cnst.output_root}sample/'
num_smpl_to_eval_on = 10_000
use_styled_conv_stylegan2 = True

flength = 5000
cam_t = np.array([0., 0., 0])
camera_params = camera_ringnetpp((512, 512), trans=cam_t, focal=flength)


run_ids_1 = [29, ]  # with sqrt(2)
# run_ids_1 = [7, 24, 8, 3]
# run_ids_1 = [7, 8, 3]

settings_for_runs = \
    {24: {'name': 'vector_cond', 'model_idx': '216000_1', 'normal_maps_as_cond': False,
          'rendered_flame_as_condition': False, 'apply_sqrt2_fac_in_eq_lin': False},
     29: {'name': 'full_model', 'model_idx': '026000_1', 'normal_maps_as_cond': True,
          'rendered_flame_as_condition': True, 'apply_sqrt2_fac_in_eq_lin': True},
     7: {'name': 'flm_rndr_tex_interp', 'model_idx': '488000_1', 'normal_maps_as_cond': False,
         'rendered_flame_as_condition': True, 'apply_sqrt2_fac_in_eq_lin': False},
     3: {'name': 'norm_mp_tex_interp', 'model_idx': '040000_1', 'normal_maps_as_cond': True,
         'rendered_flame_as_condition': False, 'apply_sqrt2_fac_in_eq_lin': False},
     8: {'name': 'norm_map_rend_flm_no_tex_interp', 'model_idx': '460000_1', 'normal_maps_as_cond': True,
         'rendered_flame_as_condition': True, 'apply_sqrt2_fac_in_eq_lin': False},}


overlay_visualizer = OverLayViz()
# overlay_visualizer.setup_renderer(mesh_file=None)

flm_params = np.zeros((num_smpl_to_eval_on, code_size)).astype('float32')
fl_param_dict = np.load(cnst.all_flame_params_file, allow_pickle=True).item()
for i, key in enumerate(fl_param_dict):
    flame_param = fl_param_dict[key]
    flame_param = np.hstack((flame_param['shape'], flame_param['exp'], flame_param['pose'], flame_param['cam'],
                             flame_param['tex'], flame_param['lit'].flatten()))
    flm_params[i, :] = flame_param.astype('float32')
    if i == num_smpl_to_eval_on - 1:
        break

batch_size = 64

fid_computer = compute_fid.FidComputer(database_root_dir=cnst.ffhq_images_root_dir,
                                       true_img_stats_dir=cnst.true_img_stats_dir)

num_sigmas = 10
corruption_sigma = np.linspace(0, 1.5, num_sigmas)
jaw_rot_range = (0, np.pi/8)
jaw_rot_sigmas = np.linspace(0, (jaw_rot_range[1] - jaw_rot_range[0])/6, num_sigmas)
pose_range = (-np.pi/3, np.pi/3)
pose_sigmas = np.linspace(0, (pose_range[1] - pose_range[0])/6, num_sigmas)
config_obj = util.dict2obj(cnst.flame_config)
flame_decoder = FLAME.FLAME(config_obj).cuda().eval()

for run_idx in run_ids_1:
    # import ipdb; ipdb.set_trace()
    generator_1 = torch.nn.DataParallel(StyledGenerator(embedding_vocab_size=69158,
                        rendered_flame_ascondition=settings_for_runs[run_idx]['rendered_flame_as_condition'],
                        normal_maps_as_cond=settings_for_runs[run_idx]['normal_maps_as_cond'],
                        core_tensor_res=core_tensor_res,
                        w_truncation_factor=1.0,
                        apply_sqrt2_fac_in_eq_lin=settings_for_runs[run_idx]['apply_sqrt2_fac_in_eq_lin'],
                        n_mlp=8)).cuda()
    model_idx = settings_for_runs[run_idx]['model_idx']
    ckpt1 = torch.load(f'{cnst.output_root}checkpoint/{run_idx}/{model_idx}.model')
    generator_1.load_state_dict(ckpt1['generator_running'])
    generator_1 = generator_1.eval()


    for i, sigma in enumerate(corruption_sigma):
        images = np.zeros((num_smpl_to_eval_on, 3, resolution, resolution)).astype('float32')
        pbar = tqdm.tqdm(range(0, num_smpl_to_eval_on, batch_size))
        pbar.set_description('Generating_images')
        flame_mesh_imgs = None
        # print(flm_params[1, :])
        for batch_idx in pbar:
            flm_batch = corrupt_flame_given_sigma(flm_params[batch_idx:batch_idx+batch_size, :],
                                                  corruption_type='all', sigma=sigma, jaw_sigma=jaw_rot_sigmas[i],
                                                  pose_sigma=pose_sigmas[i])
            flm_batch = torch.from_numpy(flm_batch).cuda()
            flm_batch = position_to_given_location(flame_decoder, flm_batch)

            if settings_for_runs[run_idx]['normal_maps_as_cond'] or \
                    settings_for_runs[run_idx]['rendered_flame_as_condition']:
                batch_size_true = flm_batch.shape[0]
                cam = flm_batch[:, constants.DECA_IDX['cam'][0]:constants.DECA_IDX['cam'][1]:]
                shape = flm_batch[:, constants.INDICES['SHAPE'][0]:constants.INDICES['SHAPE'][1]]
                exp = flm_batch[:, constants.INDICES['EXP'][0]:constants.INDICES['EXP'][1]]
                pose = flm_batch[:, constants.INDICES['POSE'][0]:constants.INDICES['POSE'][1]]
                # import ipdb; ipdb.set_trace()
                light_code = \
                    flm_batch[:, constants.DECA_IDX['lit'][0]:constants.DECA_IDX['lit'][1]:].view((batch_size_true, 9, 3))
                texture_code = flm_batch[:, constants.DECA_IDX['tex'][0]:constants.DECA_IDX['tex'][1]:]
                norma_map_img, _, _, _, rend_flm = \
                    overlay_visualizer.get_rendered_mesh(flame_params=(shape, exp, pose, light_code, texture_code),
                                                         camera_params=cam)
                rend_flm = torch.clamp(rend_flm, 0, 1) * 2 - 1
                norma_map_img = torch.clamp(norma_map_img, 0, 1) * 2 - 1
                rend_flm = fast_image_reshape(rend_flm, height_out=256, width_out=256, mode='bilinear')
                norma_map_img = fast_image_reshape(norma_map_img, height_out=256, width_out=256, mode='bilinear')

            else:
                rend_flm = None
                norma_map_img = None

            gen_1_in = ge_gen_in(flm_batch, rend_flm, norma_map_img, settings_for_runs[run_idx]['normal_maps_as_cond'],
                                 settings_for_runs[run_idx]['rendered_flame_as_condition'])

            # torch.manual_seed(2)
            identity_embeddings = torch.randint(low=0, high=69158, size=(gen_1_in.shape[0], ), dtype=torch.long,
                                                device='cuda')

            mdl_1_gen_images = generic_utils.get_images_from_flame_params(
                flame_params=gen_1_in.cpu().numpy(), pose=None,
                model=generator_1,
                step=step_max, alpha=alpha,
                input_indices=identity_embeddings.cpu().numpy())
            # import ipdb; ipdb.set_trace()
            images[batch_idx:batch_idx+batch_size_true] = torch.clamp(mdl_1_gen_images, -1, 1).cpu().numpy()
            if flame_mesh_imgs is None:
                flame_mesh_imgs = torch.clamp(norma_map_img, -1, 1).cpu().numpy()

        fid = fid_computer.get_fid(images)
        print('Fid for ' + settings_for_runs[run_idx]['name'] + ' at sigma = ' + str(sigma) + ' is: '
              + str(fid))
        if save_images:
            save_path_current_id = os.path.join(root_out_dir, str(run_idx), 'fid_with_var_sigma', f'sigma={sigma:.3f}',
                                                f'{num_smpl_to_eval_on//1000}k_fid_smpls')
            save_set_of_images(path=save_path_current_id, prefix='', images=(images[:500] + 1) / 2, show_prog_bar=True)

            #save flam rndr
            save_path_current_id_flm_rndr = os.path.join(root_out_dir, str(run_idx), 'fid_with_var_sigma',
                                                         f'sigma={sigma:.3f}',
                                                         f'{num_smpl_to_eval_on//1000}k_fid_smpls_flm_rndr')
            save_set_of_images(path=save_path_current_id_flm_rndr, prefix='', images=(flame_mesh_imgs[:50] + 1) / 2,
                               show_prog_bar=True)
