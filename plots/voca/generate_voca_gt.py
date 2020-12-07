import sys
sys.path.append('../../')
import constants as cnst
import tqdm
import numpy as np
from my_utils.visualize_flame_overlay import OverLayViz
from my_utils.generic_utils import save_set_of_images
import constants
import torch
from my_utils.eye_centering import position_to_given_location
import os
from my_utils.photometric_optimization.models import FLAME
from my_utils.photometric_optimization import util

###################################### Voca training Seq ######################################################
ignore_global_rotation = False
resolution = 256
run_idx = 29

seqs = np.load(cnst.voca_flame_seq_file)

if ignore_global_rotation:
    pose = np.hstack((seqs['frame_pose_params'][:, 0:3]*0, seqs['frame_pose_params'][:, 6:9]))
else:
    pose = np.hstack((seqs['frame_pose_params'][:, 0:3], seqs['frame_pose_params'][:, 6:9]))

num_frames = seqs['frame_exp_params'].shape[0]
translation = np.zeros((num_frames, 3))
flame_shape = np.repeat(seqs['seq_shape_params'][np.newaxis, :].astype('float32'), (num_frames,), axis=0)
flm_batch = np.hstack((flame_shape, seqs['frame_exp_params'], pose, translation)).astype('float32')[::8]
flm_batch = torch.from_numpy(flm_batch).cuda()

overlay_visualizer = OverLayViz()

config_obj = util.dict2obj(cnst.flame_config)
flame_decoder = FLAME.FLAME(config_obj).cuda().eval()
flm_batch = position_to_given_location(flame_decoder, flm_batch)


# Render FLAME
batch_size_true = flm_batch.shape[0]
cam = flm_batch[:, constants.DECA_IDX['cam'][0]:constants.DECA_IDX['cam'][1]:]
shape = flm_batch[:, constants.INDICES['SHAPE'][0]:constants.INDICES['SHAPE'][1]]
exp = flm_batch[:, constants.INDICES['EXP'][0]:constants.INDICES['EXP'][1]]
pose = flm_batch[:, constants.INDICES['POSE'][0]:constants.INDICES['POSE'][1]]
# import ipdb; ipdb.set_trace()

fl_param_dict = np.load(cnst.all_flame_params_file, allow_pickle=True).item()
random_keys = ['00000.pkl', '00001.pkl', '00002.pkl', '00003.pkl', '00004.pkl', '00005.pkl', '00006.pkl',
               '00007.pkl', '00008.pkl', '00009.pkl', '00010.pkl', '00011.pkl', '00012.pkl', '00013.pkl',
               '00014.pkl', '00015.pkl', '00016.pkl', '00017.pkl', '00018.pkl', '00019.pkl', '00020.pkl',
               '00021.pkl', '00022.pkl', '00023.pkl', '00024.pkl', '00025.pkl', '00026.pkl', '00027.pkl',
               '00028.pkl', '00029.pkl', '00030.pkl', '00031.pkl', '00032.pkl', '00033.pkl', '00034.pkl',
               '00035.pkl', '00036.pkl', '00037.pkl', '00038.pkl', '00039.pkl', '00040.pkl', '00041.pkl',
               '00042.pkl', '00043.pkl', '00044.pkl', '00045.pkl', '00046.pkl', '00047.pkl', '00048.pkl',
               '00049.pkl', '00050.pkl', '00051.pkl', '00052.pkl', '00053.pkl', '00054.pkl', '00055.pkl']

light_code = fl_param_dict[random_keys[7]]['lit'].astype('float32')[None, ...].repeat(batch_size_true, axis=0)
texture_code = fl_param_dict[random_keys[7]]['tex'].astype('float32')[None, ...].repeat(batch_size_true, axis=0)
# norma_map_img, _, _, _, rend_flm = \
#     overlay_visualizer.get_rendered_mesh(flame_params=(shape, exp, pose, torch.from_numpy(light_code).cuda(),
#                                                        torch.from_numpy(texture_code).cuda()),
#                                          camera_params=cam)


verts, landmarks2d, landmarks3d = overlay_visualizer.deca.flame(shape_params=shape, expression_params=exp,
                                                  pose_params=pose)
landmarks2d_projected = util.batch_orth_proj(landmarks2d, cam)
landmarks2d_projected[:, :, 1:] *= -1
trans_verts = util.batch_orth_proj(verts, cam)
trans_verts[:, :, 1:] = -trans_verts[:, :, 1:]

albedos = overlay_visualizer.flametex(torch.from_numpy(texture_code).cuda())
light_code = torch.from_numpy(light_code).cuda()
rendering_results = overlay_visualizer.deca.render(verts, trans_verts, albedos, lights=light_code, light_type='point',
                                                   cull_backfaces=True)
textured_images, normals, alpha_img = rendering_results['images'], rendering_results['normals'],\
                                      rendering_results['alpha_images']
normal_images = overlay_visualizer.deca.render.render_normal(trans_verts, normals)


rend_flm = torch.clamp(textured_images, 0, 1) * 2 - 1
# rend_flm += (1 - alpha_img) * 2
norma_map_img = torch.clamp(normal_images, 0, 1) * 2 - 1
# norma_map_img += (1 - alpha_img) * 2


for id in tqdm.tqdm(range(50)):
    save_set_of_images(path=os.path.join(cnst.output_root, f'sample/{run_idx}/voca/'+str(id)), prefix='mesh_normal_',
                       images=((norma_map_img + 1) / 2).cpu().numpy())

    save_set_of_images(path=os.path.join(cnst.output_root, f'sample/{run_idx}/voca/' + str(id)), prefix='mesh_textured_',
                       images=((rend_flm + 1) / 2).cpu().numpy())

    # save_set_of_images(path=os.path.join(cnst.output_root, f'sample/{run_idx}/voca/'+str(id)), prefix='',
    #                    images=((fake_images + 1) / 2))
print(f'Voca Animation saved to {os.path.join(cnst.output_root, f"sample/{run_idx}/voca/")}')
