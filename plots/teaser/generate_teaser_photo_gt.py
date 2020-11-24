import sys
sys.path.append('../../')
import numpy as np
import torch
from dataset_loaders import fast_image_reshape
from model.stg2_generator import StyledGenerator
import os
import constants as cnst
from my_utils.visualize_flame_overlay import OverLayViz
from my_utils.flm_dynamic_fit_overlay import camera_ringnetpp, camera_dynamic
from my_utils.generic_utils import save_set_of_images
import glob
from my_utils.eye_centering import position_to_given_location
from model.mesh_and_3d_helpers import batch_orth_proj


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
flame_version = 'DECA'
with_neck = False
random_background = False
texture_pattern = 'MEAN_TEXTURE_WITH_CHKR_BOARD'
use_inst_norm = True
normal_maps_as_cond = True
use_posed_constant_input = False
core_tensor_res = 4
run_id = 29
model_idx = '026000_1'
ckpt = torch.load(f'{cnst.output_root}/checkpoint/{run_id}/{model_idx}.model')
other_params = np.load(f'{cnst.output_root}/checkpoint/{run_id}/{model_idx}.npz')
alpha = float(other_params['alpha'])
alpha = 1
normalization_file_path = f'{cnst.output_root}/FFHQ_dynamicfit_normalization.npz'
normalization_file_path = f'{cnst.output_root}/FFHQ_dynamicfit_normalization_hawen_parms.npz'

if 'resolution' in other_params.files:
    resolution = other_params['resolution']

rendered_flame_as_condition = True
rows = 5
cols = 6
b_size = rows*cols
n_frames = 32
step_max = int(np.log2(resolution) - 2)  # starts from 4X4 hence the -2

torch.manual_seed(7)
generator = StyledGenerator(embedding_vocab_size=69158,
                        rendered_flame_ascondition=True,
                        normal_maps_as_cond=True,
                        core_tensor_res=core_tensor_res,
                        w_truncation_factor=1.0,
                        apply_sqrt2_fac_in_eq_lin=True,
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


# Load FFHQ all FLAME params. For centring
num_smpl_to_eval_on = 65000
# num_smpl_to_eval_on = 600
flm_params_ffhq_all = np.zeros((num_smpl_to_eval_on, code_size)).astype('float32')
fl_param_dict = np.load(cnst.all_flame_params_file, allow_pickle=True).item()
list_bad_images = np.load(cnst.list_deca_failed_iamges)['bad_images']


random_light_texture_code_idx = 3
count = 0
for i, key in enumerate(fl_param_dict):
    flame_param = fl_param_dict[key]
    flame_param_flttenned = np.hstack((flame_param['shape'], flame_param['exp'], flame_param['pose'],
                                       flame_param['cam']))
    flm_params_ffhq_all[count, :] = flame_param_flttenned.astype('float32')
    count += 1
    if i in list_bad_images:
        continue
    if i == random_light_texture_code_idx:
        light_code, texture_code = flame_param['lit'], flame_param['tex']
    if i == num_smpl_to_eval_on - 1:
        break

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
mean_minus_3_sigma_light = mean_lighting
mean_minus_3_sigma_light[0, highest_variance_cmp_idx] -= 2*highest_variance
mean_minus_3_sigma_light = torch.from_numpy(mean_minus_3_sigma_light.astype('float32')).cuda()

flm_params = torch.cat(flm_params, dim=0)
flame_decoder = overlay_visualizer.deca.flame.eval()
flm_params = position_to_given_location(flame_decoder, flm_params)


cam = flm_params[:, cnst.DECA_IDX['cam'][0]:cnst.DECA_IDX['cam'][1]:]
shape = flm_params[:, cnst.INDICES['SHAPE'][0]:cnst.INDICES['SHAPE'][1]]
exp = flm_params[:, cnst.INDICES['EXP'][0]:cnst.INDICES['EXP'][1]]
pose = flm_params[:, cnst.INDICES['POSE'][0]:cnst.INDICES['POSE'][1]]
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

verts, landmarks2d, landmarks3d = overlay_visualizer.deca.flame(shape_params=shape, expression_params=exp,
                                                  pose_params=pose)
landmarks2d_projected = batch_orth_proj(landmarks2d, cam)
landmarks2d_projected[:, :, 1:] *= -1
trans_verts = batch_orth_proj(verts, cam)
trans_verts[:, :, 1:] = -trans_verts[:, :, 1:]

right_albedos = overlay_visualizer.flametex(texture_code)
# albedos = torch.tensor([47, 59, 65], dtype=torch.float32)[None, ..., None, None].cuda()/255.0*1.5
albedos = torch.tensor([0.6, 0.6, 0.6], dtype=torch.float32)[None, ..., None, None].cuda()
albedos = albedos.repeat(texture_code.shape[0], 1, 512, 512)
albedos[-4:] = fast_image_reshape(right_albedos[-4:], height_out=512, width_out=512)

rendering_results = overlay_visualizer.deca.render(verts, trans_verts, albedos, lights=light_code, light_type='point',
                                                   cull_backfaces=True)
textured_images, normals, alpha_img = rendering_results['images'], rendering_results['normals'],\
                                      rendering_results['alpha_images']
normal_images = overlay_visualizer.deca.render.render_normal(trans_verts, normals)


rend_flm = torch.clamp(textured_images, 0, 1) * 2 - 1
norma_map_img = torch.clamp(normal_images, 0, 1) * 2 - 1

id_start = 20
save_dir_teaser = os.path.join(save_dir_tsr, f'images_gt_FLAME/')
os.makedirs(save_dir_teaser, exist_ok=True)


rend_flm_copy = rend_flm.clone()
rend_flm_copy += (1 - alpha_img) * 2
save_set_of_images(path=save_dir_teaser, prefix='mesh_textured_',
                   images=((rend_flm_copy + 1) / 2).cpu().numpy(),
                   name_list=file_names)
