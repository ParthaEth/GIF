import tqdm
from my_utils.flm_dynamic_fit_overlay import camera_ringnetpp
import numpy as np
from my_utils.visualize_flame_overlay import OverLayViz
from my_utils.ringnet_overlay.util import tensor_vis_landmarks
from PIL import Image
import torch
import matplotlib.pyplot as plt


resolution = 512
flength = 5000
cam_t = np.array([0., 0., 0])
camera_params = camera_ringnetpp((resolution, resolution), trans=cam_t, focal=flength)

overlay_visualizer = OverLayViz(full_neck=False, add_random_noise_to_background=False,
                                background_img=None, texture_pattern_name='CHKR_BRD_FLT_TEETH',
                                flame_version='FLAME_2020_revisited', image_size=resolution)

dest_dir = '/is/cluster/work/pghosh/gif1.0/fitting_viz/flame_2020/'
# dest_dir = '/is/cluster/work/pghosh/gif1.0/fitting_viz/flame_old/'

for i in tqdm.tqdm(range(0, 300)):
    img_file = '/is/cluster/scratch/partha/face_gan_data/FFHQ/images1024x1024/' + str(i).zfill(5) + '.png'
    img_original = Image.open(img_file).resize((resolution, resolution))
    images = torch.from_numpy(np.array(img_original).transpose((2, 0, 1)).astype('float32'))[None, ...].cuda()

    flame_param_file = f'/is/cluster/pghosh/face_gan_data/FFHQ/flmae_photometric_opt/' \
                       f'faceHQ_cleaned_40k_bfm50_skin_mask_2stages_FLAME_2020_revisited_np/{str(i).zfill(5)}.npy'
    try:
        flame_param = np.load(flame_param_file, allow_pickle=True)
    except FileNotFoundError as e:
        continue

    flame_param = np.hstack((flame_param['shape'], flame_param['exp'], flame_param['pose'], flame_param['cam']))

    tz = flength / (0.5 * resolution * flame_param[:, 156:157])
    flame_param[:, 156:159] = np.concatenate((flame_param[:, 157:], tz), axis=1)

    flame_param = torch.from_numpy(flame_param).cuda()

    norma_map_img, pos_mask, alpha_images, key_points2d, rend_imgs = \
        overlay_visualizer.get_rendered_mesh(
            flame_params=(flame_param[:, 0:100],  # shape
                          flame_param[:, 100:150],  # exp
                          flame_param[:, 150:156],  # Pose
                          flame_param[:, 156:159]),  # translation
            camera_params=camera_params)

    overlayed_imgs_original = tensor_vis_landmarks(images/255, key_points2d[:, 17:])
    plt.imshow(overlayed_imgs_original[0].cpu().detach().numpy().transpose((1, 2, 0)))
    print('Shown')
    plt.show()
    # img_to_save = (255*overlayed_imgs_original[0]).cpu().detach().numpy().transpose((1, 2, 0)).astype('uint8')
    # img_to_save = Image.fromarray(img_to_save)
    # img_to_save.save(dest_dir + str(i) + '.png')