import constants as cnst
import tqdm
from model.stg2_generator import StyledGenerator
import numpy as np
from my_utils.visualize_flame_overlay import OverLayViz
from my_utils.generic_utils import save_set_of_images
import constants
from dataset_loaders import fast_image_reshape
import torch
from my_utils import generic_utils
from my_utils.eye_centering import position_to_given_location
import os
import argparse
from configurations import update_config

###################################### Voca training Seq ######################################################
ignore_global_rotation = False
resolution = 256
run_idx = 29

parser = argparse.ArgumentParser(description='Progressive Growing of GANs')
args, dataset, flame_param_est = update_config(parser)

generator_1 = torch.nn.DataParallel(StyledGenerator(embedding_vocab_size=args.embedding_vocab_size,
                                                    rendered_flame_ascondition=args.rendered_flame_as_condition,
                                                    normal_maps_as_cond=args.normal_maps_as_cond,
                                                    core_tensor_res=args.core_tensor_res,
                                                    n_mlp=args.nmlp_for_z_to_w)).cuda()
model_idx = '026000_1'
ckpt1 = torch.load(f'{cnst.output_root}checkpoint/{run_idx}/{model_idx}.model')
generator_1.load_state_dict(ckpt1['generator_running'])
generator_1 = generator_1.eval()

seqs = np.load(cnst.voca_flame_seq_file)

if ignore_global_rotation:
    pose = np.hstack((seqs['frame_pose_params'][:, 0:3]*0, seqs['frame_pose_params'][:, 6:9]))
else:
    pose = np.hstack((seqs['frame_pose_params'][:, 0:3], seqs['frame_pose_params'][:, 6:9]))

num_frames = seqs['frame_exp_params'].shape[0]
translation = np.zeros((num_frames, 3))
shape_seq = seqs['seq_shape_params']
shape_seq[3:] *= 0
flame_shape = np.repeat(shape_seq[np.newaxis, :].astype('float32'), (num_frames,), axis=0)
flm_batch = np.hstack((flame_shape, seqs['frame_exp_params'], pose, translation)).astype('float32')
flm_batch = torch.from_numpy(flm_batch).cuda()

overlay_visualizer = OverLayViz(full_neck=False, add_random_noise_to_background=False, inside_mouth_faces=True,
                                background_img=None, texture_pattern_name='MEAN_TEXTURE_WITH_CHKR_BOARD',
                                flame_version='DECA', image_size=256)

flame_decoder = overlay_visualizer.deca.flame.eval()
flm_batch = position_to_given_location(flame_decoder, flm_batch)


# Render FLAME
seq_len = flm_batch.shape[0]
cam = flm_batch[:, constants.DECA_IDX['cam'][0]:constants.DECA_IDX['cam'][1]:]
shape = flm_batch[:, constants.INDICES['SHAPE'][0]:constants.INDICES['SHAPE'][1]]
exp = flm_batch[:, constants.INDICES['EXP'][0]:constants.INDICES['EXP'][1]]
pose = flm_batch[:, constants.INDICES['POSE'][0]:constants.INDICES['POSE'][1]]
# import ipdb; ipdb.set_trace()

light_texture_id_code_source = np.load('../teaser/params.npy', allow_pickle=True).item()
ids_to_pick = [1024,  1467,  1552,  1614,  1904,  238,  327,  495,
               1069,  1468,  1554,  1642,  1914,  259,  355,  663,
               127,   1471,  1565,  1683,  1947,  261,  356,
               1427,  1472,  1571,  1891,  2047,  309,  48,]


for id in tqdm.tqdm(ids_to_pick):
    fake_images = []
    norma_map_img = []
    rend_flm = []

    for batch_idx in range(0, seq_len, 32):
        shape_batch = shape[batch_idx:batch_idx+32]
        exp_batch = exp[batch_idx:batch_idx+32]
        pose_batch = pose[batch_idx:batch_idx+32]
        cam_batch = cam[batch_idx:batch_idx+32]
        true_batch_size = cam_batch.shape[0]
        light_code = light_texture_id_code_source['light_code'][id].astype('float32')[None, ...].\
            repeat(true_batch_size, axis=0)
        texture_code = light_texture_id_code_source['texture_code'][id].astype('float32')[None, ...].\
            repeat(true_batch_size, axis=0)
        # import ipdb; ipdb.set_trace()
        norma_map_img_batch, _, _, _, rend_flm_batch = \
            overlay_visualizer.get_rendered_mesh(flame_params=(shape_batch, exp_batch, pose_batch,
                                                               torch.from_numpy(light_code).cuda(),
                                                               torch.from_numpy(texture_code).cuda()),
                                                 camera_params=cam_batch)
        rend_flm_batch = torch.clamp(rend_flm_batch, 0, 1) * 2 - 1
        norma_map_img_batch = torch.clamp(norma_map_img_batch, 0, 1) * 2 - 1
        rend_flm_batch = fast_image_reshape(rend_flm_batch, height_out=256, width_out=256, mode='bilinear')
        norma_map_img_batch = fast_image_reshape(norma_map_img_batch, height_out=256, width_out=256, mode='bilinear')

        # with back face culling and white texture
        norma_map_img_batch_to_save, _, _, _, rend_flm_batch_to_save = \
            overlay_visualizer.get_rendered_mesh(flame_params=(shape_batch, exp_batch, pose_batch,
                                                               torch.from_numpy(light_code).cuda(),
                                                               torch.from_numpy(texture_code).cuda()),
                                                 camera_params=cam_batch, cull_backfaces=True,
                                                 grey_texture=True)
        rend_flm_batch_to_save = torch.clamp(rend_flm_batch_to_save, 0, 1) * 2 - 1
        norma_map_img_batch_to_save = torch.clamp(norma_map_img_batch_to_save, 0, 1) * 2 - 1
        rend_flm_batch_to_save = fast_image_reshape(rend_flm_batch_to_save, height_out=256, width_out=256, mode='bilinear')
        norma_map_img_batch_to_save = fast_image_reshape(norma_map_img_batch_to_save, height_out=256, width_out=256, mode='bilinear')

        gen_in = torch.cat((rend_flm_batch, norma_map_img_batch), dim=1)

        identity_embeddings = (np.ones((true_batch_size, )) *
                               int(light_texture_id_code_source['identity_indices'][id])).astype('int64')
        fake_images_batch = generic_utils.get_images_from_flame_params(
            flame_params=gen_in.cpu().numpy(), pose=None,
            model=generator_1,
            step=int(np.log2(resolution) - 2), alpha=1.0,
            input_indices=identity_embeddings)
        # import ipdb; ipdb.set_trace()
        fake_images_batch = torch.clamp(fake_images_batch, -1, 1)

        fake_images.append(fake_images_batch.cpu())
        rend_flm.append(rend_flm_batch_to_save.cpu())
        norma_map_img.append(norma_map_img_batch_to_save.cpu())

    fake_images = torch.cat(fake_images, dim=0)
    rend_flm = torch.cat(rend_flm, dim=0)
    norma_map_img = torch.cat(norma_map_img, dim=0)


    save_set_of_images(path=os.path.join(cnst.output_root, f'sample/{run_idx}/voca/selected_ids_'+str(id)),
                       prefix='mesh_normal_', images=((norma_map_img + 1) / 2).numpy())

    save_set_of_images(path=os.path.join(cnst.output_root, f'sample/{run_idx}/voca/selected_ids_' + str(id)),
                       prefix='mesh_textured_', images=((rend_flm + 1) / 2).numpy())

    save_set_of_images(path=os.path.join(cnst.output_root, f'sample/{run_idx}/voca/selected_ids_'+str(id)), prefix='',
                       images=((fake_images + 1) / 2).numpy())
print(f'Voca Animation saved to {os.path.join(cnst.output_root, f"sample/{run_idx}/voca/")}')
