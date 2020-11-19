import numpy as np
import glob
import os
import random
import tqdm
import torch
from torchvision import transforms, utils
from PIL import Image
from skvideo.io import vwrite
import imageio
import pickle
from copy import deepcopy
from my_utils.visualize_flame_overlay import OverLayViz
from my_utils.ringnet_overlay.util import tensor_vis_landmarks
from dataset_loaders import fast_image_reshape
import constants
from my_utils.flm_dynamic_fit_overlay import camera_dynamic
import torchvision

# # useful constant
SHAPE_IDS = constants.INDICES['SHAPE']
EXP_IDS = constants.INDICES['EXP']
POSE_IDS = constants.INDICES['POSE']
TRANS_IDS = constants.INDICES['TRANS']
# CAM_IDS = constants.INDICES['CAM']
# JAW_ROT_IDS = INDICES['JAW_ROT']
# GLOBAL_ROT_IDS = INDICES['GLOBAL_ROT']
# ROT_JAW_CAM = INDICES['ROT_JAW_CAM']

flame_params_path = '../../New_Flame.txt._is_scratch'


def get_original_images(flame_path):
    image_path = flame_path.replace('/New_flame/', '/Cropped_frames/')
    return Image.open(image_path.replace('npy', 'png'))


def read_flame_param(flame_path, data_type = 'VOXCELEB'):
    '''
        Return flame parameters: [0:100 - Shape, 100:150 - Expression, 150:156 - Pose, 156:159 - Camera]
    '''
    if data_type == 'VOXCELEB':
        flame_path_old = flame_path.replace('/New_flame/', '/Flame_params/')

        flame_new = np.load(flame_path)
        flame_old = np.load(flame_path_old)

        params = flame_old[:]
        # params[0:100] = np.array(shape_params[id])
        params[100:150] = flame_new[:50]
        params[153:156] = flame_new[53:56]

        # TODO: SEND 156 dimensional param vector where params[:100] corresponds to shape
        params = params[:156]

    elif data_type == 'FFHQ':
        with open(flame_path, 'rb') as param_file:
            flame_param = pickle.load(param_file, encoding='latin1')

        params = np.hstack((flame_param['betas'][:100], flame_param['betas'][300:350], flame_param['pose'][0:3],
                                 flame_param['pose'][6:9]))

    return params


def make_flame_shape_consitant(flame_params):
     flame_median = np.median(flame_params[:,:100], axis=0)
     flame_params[:, :100] = flame_median
     return flame_params


def save_images_to_gif(images, save_path):
    '''
        Input: images: Batch_Size x 3 x H x W [Torch.Tensor]
    '''
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    images = images.clamp(min=0, max=1.0)
    imgs = images.detach().cpu().numpy().transpose((0, 2, 3, 1)) * 255
    imgs = imgs.astype('uint8')

    imageio.imwrite(save_path.replace('.gif', '.png'), imgs[0])
    imageio.mimsave(save_path, imgs)

    vwrite(save_path.replace('.gif', '.mp4'), imgs, outputdict={'-r': '30'})


def save_images_with_flame_to_gif(images, save_path):
    pass
    '''
        Input: images: Batch_Size x 3 x H x W [Torch.Tensor]
    '''
    # batch_size = images.shape[0]
    # frames = []
    #
    # for img in images:
    #     if isinstance(img, Image.Image):
    #         new_frame = img
    #     else:
    #         new_frame = transforms.ToPILImage()((img.cpu().data + 1)/2)
    #     frames.append(new_frame)
    #
    # frames[0].save(save_path, format='GIF', append_images=frames[1:], save_all=True, duration=41, loop=0)


def generate_from_flame_sequence(generator, gen_in, pose, step, alpha, noise, input_indices):
    '''
        Input: flame_params: Batch_Size x Flame_Code_Size
        Output: fake_images: Batch_Size x 3 x H x W
    '''
    max_batch_szie = 32
    generator.eval()

    images = []
    with torch.no_grad():
        for batch_idx in range(0, gen_in.shape[0], max_batch_szie):
            gen_in_this_batch = gen_in[batch_idx:batch_idx + max_batch_szie]

            if type(gen_in) is np.ndarray:
                gen_in_this_batch = torch.from_numpy(gen_in_this_batch).cuda()

            if pose is not None and type(pose) is np.ndarray:
                pose_this_batch = torch.from_numpy(pose[batch_idx:batch_idx + max_batch_szie]).cuda()
            else:
                pose_this_batch = None

            if input_indices.shape[0] == 1:
                identities = input_indices.repeat((gen_in_this_batch.shape[0],))
            else:
                identities = input_indices[batch_idx:batch_idx + max_batch_szie]

            fake_images = generator(gen_in_this_batch, pose_this_batch, step=step, alpha=alpha, noise=noise,
                                    input_indices=identities)
            images.append(fake_images[-1].cpu())

    if len(images) > 1:
        images = [torch.cat(images, dim=0)]

    return images


def generate_gif_from_vgg_params(generator, flame_params_path, normalization_file_path, log_dir, step, noise, n_rows=5,
                                 n_cols=4, seed=0, n_frames=8, dump_original=False):
    random.seed(seed)
    os.makedirs(log_dir, exist_ok=True)

    grid_size = n_rows * n_cols
    with open(flame_params_path, 'r') as f:
        flame_param_files = f.readlines()

    if normalization_file_path is not None:
        normalization_data = np.load(normalization_file_path)
        flame_mean = normalization_data['mean'][0]
        flame_std = normalization_data['std'][0]

    # pick N=grid_size different random sequence
    cell_ids = random.sample(flame_param_files, 50)

    for i, cell in enumerate(tqdm.tqdm(cell_ids)):
        cell_dir = os.path.dirname(cell)
        flame_files = sorted(glob.glob(os.path.join(cell_dir, '*.npy')))[:n_frames]
        assert len(flame_files) >= n_frames, 'Number of video frames are' + str(len(flame_files))

        flame_params = np.array(list(map(read_flame_param, flame_files)))
        flame_params = make_flame_shape_consitant(flame_params)

        # Keep only strongly meaningful dimensions
        # flame_params[:, :100] = 0
        # flame_params[:, 100:150] = 0
        # flame_params[153:156] = 0
        # flame_params[153:156]

        # Normalization
        if normalization_file_path is not None:
            flame_params = (flame_params - flame_mean)/flame_std

        fake_images = generate_from_flame_sequence(generator, flame_params, step, noise)[-1]

        save_path = os.path.join(log_dir, str(i).zfill(4)+'.gif')
        save_images_to_gif(fake_images, save_path)

        if dump_original:
            path_dirs = save_path.split('/')
            save_path_original = path_dirs[:-1] + ['original'] + [str(i).zfill(4)+'.gif']
            original_images = list(map(get_original_images, flame_files))
            save_images_to_gif(original_images, os.path.join(*save_path_original))


def get_gif_from_list_of_params(generator, flame_params, step, alpha, noise, overlay_landmarks, flame_std, flame_mean,
                                overlay_visualizer, rendered_flame_as_condition, use_posed_constant_input,
                                normal_maps_as_cond, camera_params):
    # cam_t = np.array([0., 0., 2.5])
    # camera_params = camera_dynamic((224, 224), cam_t)
    if overlay_visualizer is None:
        overlay_visualizer = OverLayViz(add_random_noise_to_background=False)

    fixed_embeddings = torch.ones(flame_params.shape[0], dtype=torch.long, device='cuda')*13
    # print(generator.module.get_embddings()[fixed_embeddings])
    flame_params_unnorm = flame_params * flame_std + flame_mean

    flame_params_unnorm = torch.from_numpy(flame_params_unnorm).cuda()
    normal_map_img, pos_mask, alpha_images, key_points2d, rend_imgs = \
        overlay_visualizer.get_rendered_mesh(flame_params=(flame_params_unnorm[:, SHAPE_IDS[0]:SHAPE_IDS[1]],
                                                           flame_params_unnorm[:, EXP_IDS[0]:EXP_IDS[1]],
                                                           flame_params_unnorm[:, POSE_IDS[0]:POSE_IDS[1]],
                                                           flame_params_unnorm[:, TRANS_IDS[0]:TRANS_IDS[1]]),
                                             camera_params=camera_params)
    rend_imgs = (rend_imgs/127.0 - 1)

    if use_posed_constant_input:
        pose = flame_params[:, constants.get_idx_list('GLOBAL_ROT')]
    else:
        pose = None

    if rendered_flame_as_condition:
        gen_in = rend_imgs
    else:
        gen_in = flame_params

    if normal_maps_as_cond:
        gen_in = torch.cat((rend_imgs, normal_map_img), dim=1)

    fake_images = generate_from_flame_sequence(generator, gen_in, pose, step, alpha, noise,
                                               input_indices=fixed_embeddings)[-1]

    fake_images = overlay_visualizer.range_normalize_images(fast_image_reshape(fake_images,
                                                                               height_out=rend_imgs.shape[2],
                                                                               width_out=rend_imgs.shape[3]))
    if overlay_landmarks:
        if key_points2d.shape[-2] > 68:
            key_points2d = key_points2d[:, 17:, :]
        fake_images_overlay = tensor_vis_landmarks(fake_images, key_points2d)
        fake_images = torch.cat([fake_images.cpu(), fake_images_overlay], dim=-1)

    if rendered_flame_as_condition:
        fake_images = torch.cat([fake_images.cpu(), (rend_imgs.cpu() + 1)/2], dim=-1)

    if normal_maps_as_cond:
        fake_images = torch.cat([fake_images.cpu(), (normal_map_img.cpu() + 1) / 2], dim=-1)

    return fake_images


def interpolate_FFHQ(generator, flame_param_paths, normalization_file_path, log_dir, step, alpha, noise, camera_params,
                     seed=0, n_frames=8, flame_params=None, overlay_landmarks=False, flame_mean=None, flame_std=None,
                     rendered_flame_as_condition=False, overlay_visualizer=None, use_posed_constant_input=False,
                     normal_maps_as_cond=False):
    '''
        cell1: Vary shape
        cell2: Vary expression
        cell3: Vary Global Pose
        cell4: Vary Jaw Pose
    '''

    random.seed(seed)
    os.makedirs(log_dir, exist_ok=True)

    if flame_params is None:
        flame_param1 = read_flame_param(flame_param_paths[0], data_type='FFHQ')
        flame_param2 = read_flame_param(flame_param_paths[1], data_type='FFHQ')

        if normalization_file_path is not None:
            normalization_data = np.load(normalization_file_path)
            flame_mean = normalization_data['mean'][0]
            flame_std = normalization_data['std'][0]
            flame_param1 = (flame_param1 - flame_mean)/flame_std
            flame_param2 = (flame_param2 - flame_mean)/flame_std
    else:
        flame_param1, flame_param2 = (flame_params - flame_mean)/flame_std

    # cell_ids = [SHAPE_IDS, EXP_IDS, GLOBAL_ROT_IDS, JAW_ROT_IDS, ROT_JAW_CAM]
    # cell_names = ['SHAPE', 'EXP', 'GLOBAL_ROT', 'JAW_ROT', ('GLOBAL_ROT', 'JAW_ROT'), ('JAW_ROT', 'EXP'), 'TRANS',
    #               ('GLOBAL_ROT', 'TRANS'), ('SHAPE', 'GLOBAL_ROT', 'TRANS')]
    cell_names = [('JAW_ROT', 'EXP'),  ('GLOBAL_ROT', 'TRANS')]

    start_end_images = get_gif_from_list_of_params(generator, np.vstack((flame_param1, flame_param2)), step, alpha,
                                                   noise, overlay_landmarks, flame_std, flame_mean, overlay_visualizer,
                                                   rendered_flame_as_condition, use_posed_constant_input,
                                                   normal_maps_as_cond, camera_params)
    torchvision.utils.save_image(
        start_end_images,
        os.path.join(log_dir, 'start_end_overlayed.png'),
        nrow=2,
        normalize=True,
        range=(0, 1))

    for cell_name in tqdm.tqdm(cell_names):
        # cell_dir = os.path.dirname(cell)
        # flame_files = sorted(glob.glob(os.path.join(cell_dir, '*.npy')))[:n_frames]
        # assert len(flame_files) >= n_frames, 'Number of video frames are' + str(len(flame_files))
        cell_idxs = constants.get_idx_list(cell_name)
        flame_params = np.expand_dims(deepcopy(flame_param1), axis=0).repeat(n_frames, axis=0).astype('float32')
        # import ipdb; ipdb.set_trace()
        flame_params[:, cell_idxs] = np.linspace(flame_param1[cell_idxs], flame_param2[cell_idxs], n_frames,
                                                    endpoint=True)

        # # hold some flame prams constant
        # flame_params[:, 5:100] = flame_params[0, 5:100]
        # flame_params[:, 105:150] = flame_params[0, 105:150]
        # flame_params *= 0

        fake_images = get_gif_from_list_of_params(generator, flame_params, step, alpha, noise, overlay_landmarks,
                                                  flame_std, flame_mean, overlay_visualizer,
                                                  rendered_flame_as_condition, use_posed_constant_input,
                                                  normal_maps_as_cond, camera_params)
        # fake_images = fake_images.clamp_(0, 1)

        if type(cell_name) == tuple:
            save_name = ''
            for name_frg in cell_name:
                save_name += (name_frg + '_')
        else:
            save_name = cell_name

        save_path = os.path.join(log_dir, save_name + '.gif')
        # save_images_with_flame_to_gif(fake_images, flame, save_path)
        save_images_to_gif(fake_images, save_path)