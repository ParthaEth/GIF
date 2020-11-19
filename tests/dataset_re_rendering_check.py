import sys
sys.path.append('../')
import os
from model import Blur
from dataset_loaders import FFHQ, fast_image_reshape
from torchvision import transforms
import matplotlib.pyplot as plt
import random
import numpy as np
import constants
import torch
from my_utils.visualize_flame_overlay import OverLayViz

generic_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),])


params_dir = '/is/cluster/work/pghosh/gif1.0/DECA_inferred/deca_flame_params_camera_corrected.npy'
data_root = '/is/cluster/pghosh/face_gan_data/FFHQ/multiscale/'
rendered_flame_root = '/is/cluster/work/pghosh/gif1.0/DECA_inferred/deca_rendered_with_texture.lmdb'
normalization_file_path = None
flame_param_est = None

# import ipdb; ipdb.set_trace()
list_bad_images = np.load('/is/cluster/work/pghosh/gif1.0/DECA_inferred/b_box_stats.npz')['bad_images']
dataset = FFHQ(real_img_root=data_root, rendered_flame_root=rendered_flame_root, params_dir=params_dir,
               generic_transform=generic_transform, pose_cam_from_yao=False,
               rendered_flame_as_condition=True, resolution=256,
               normalization_file_path=normalization_file_path, debug=True, random_crop=False, get_normal_images=True,
               flame_version='DECA', list_bad_images=list_bad_images)

show_img_at_res = 256

overlay_visualizer = OverLayViz(full_neck=False, add_random_noise_to_background=False, inside_mouth_faces=True,
                                background_img=None, texture_pattern_name='MEAN_TEXTURE_WITH_CHKR_BOARD',
                                flame_version='DECA', image_size=256)

example_out_dir = '/is/cluster/work/pghosh/gif1.0/DECA_inferred/saved_rerendered_side_by_side'

for indx in range(5):
    # i = int(random.randint(0, 60_000))
    i = indx

    fig, (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8) = plt.subplots(1, 8)
    img, flm_rndr, flm_lbl, index = dataset.__getitem__(i, bypass_valid_indexing=False)
    img = img[None, ...]
    flm_rndr = flm_rndr[0][None, ...]

    img = fast_image_reshape(img, height_out=show_img_at_res, width_out=show_img_at_res)
    flm_rndr = fast_image_reshape(flm_rndr, height_out=show_img_at_res, width_out=show_img_at_res)

    img = img[0]
    flm_rndr = flm_rndr[0]

    flm_rndr = (flm_rndr.numpy().transpose((1, 2, 0)) + 1)/2
    ax1.imshow((img.numpy().transpose((1, 2, 0)) + 1)/2)
    ax2.imshow(flm_rndr[:, :, :3])

    if flm_rndr.shape[-1] > 3:
        ax3.imshow(flm_rndr[:, :, 3:])

    # Rerendering
    # import ipdb; ipdb.set_trace()
    flm_lbl = torch.from_numpy(flm_lbl[0][None, ...]).cuda()
    cam = flm_lbl[:, constants.DECA_IDX['cam'][0]:constants.DECA_IDX['cam'][1]:]
    shape = flm_lbl[:, constants.INDICES['SHAPE'][0]:constants.INDICES['SHAPE'][1]]
    exp = flm_lbl[:, constants.INDICES['EXP'][0]:constants.INDICES['EXP'][1]]
    pose = flm_lbl[:, constants.INDICES['POSE'][0]:constants.INDICES['POSE'][1]]
    texture_code = flm_lbl[:, constants.DECA_IDX['tex'][0]:constants.DECA_IDX['tex'][1]]
    light_code = flm_lbl[:, constants.DECA_IDX['lit'][0]:constants.DECA_IDX['lit'][1]].reshape((-1, 9, 3))

    norma_map_img, _, _, _, rend_flm = \
        overlay_visualizer.get_rendered_mesh(flame_params=(shape, exp, pose, light_code, texture_code),
                                             camera_params=cam)
    # import ipdb; ipdb.set_trace()
    rend_flm = fast_image_reshape(rend_flm, height_out=show_img_at_res, width_out=show_img_at_res, mode='bilinear')
    norma_map_img = fast_image_reshape(norma_map_img, height_out=show_img_at_res, width_out=show_img_at_res,
                                       mode='bilinear')
    rend_flm = torch.clamp(rend_flm, 0, 1) * 2 - 1
    norma_map_img = torch.clamp(norma_map_img, 0, 1) * 2 - 1

    rend_flm = (rend_flm.detach().cpu().numpy()[0].transpose((1, 2, 0)) + 1)/2
    ax4.imshow(rend_flm)

    norma_map_img = (norma_map_img.detach().cpu().numpy()[0].transpose((1, 2, 0)) + 1) / 2
    ax5.imshow(norma_map_img)

    ax6.imshow(flm_rndr[:, :, :3] - rend_flm)
    ax7.imshow(flm_rndr[:, :, 3:] - norma_map_img)
    import ipdb; ipdb.set_trace()
    # look at (flm_rndr[:, :, :3] - rend_flm).max() and (flm_rndr[:, :, :3] - rend_flm).min() must be close to zero

    plt.savefig(os.path.join(example_out_dir, str(i)+'.png'))