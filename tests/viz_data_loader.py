import sys
sys.path.append('../')
import os
from model import Blur
from dataset_loaders import FFHQ, fast_image_reshape
from torchvision import transforms
import matplotlib.pyplot as plt
import random
import numpy as np

generic_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),])


# params_dir = '/raid/data/pghosh/face_gan_data/flame_dynamic/params_locked_neck/'
# params_dir = '/is/cluster/pghosh/face_gan_data/FFHQ/params_locked_neck'
# params_dir = '/is/cluster/pghosh/face_gan_data/FFHQ/flame_dynamic_2020.npy'
params_dir = '/is/cluster/work/pghosh/gif1.0/DECA_inferred/deca_flame_params.npy'
# params_dir = '/is/cluster/pghosh/face_gan_data/FFHQ/ffhq_fixed_neck.npy'
# data_root = '/raid/data/pghosh/face_gan_data/FFHQ/multiscale/'
# data_root = '/is/cluster/pghosh/face_gan_data/FFHQ/multiscale/'
data_root = '/is/cluster/work/pghosh/gif1.0/FFHQ/'
# rendered_flame_root = '/is/cluster/pghosh/face_gan_data/FFHQ/rendered_flame_no_neck_rand_bg.lmdb'
# rendered_flame_root = '/is/cluster/pghosh/face_gan_data/FFHQ/rendered_flame_no_neck_rand_bg.lmdb'
# rendered_flame_root = '/is/cluster/work/pghosh/gif1.0/rendered_flame_no_neck_black_bg.lmdb'
# rendered_flame_root = '/is/cluster/work/pghosh/gif1.0/rendered_flame_no_neck_random_bg_chkr.lmdb'
# rendered_flame_root = '/is/cluster/work/pghosh/gif1.0/rendered_flame_no_neck_black_bg_chk_texture.lmdb'
# rendered_flame_root = '/is/cluster/pghosh/face_gan_data/FFHQ/rendered_flame_no_neck_black_bg_chk_texture_flt_teeth_color_norm_map.lmdb'
# rendered_flame_root = '/is/cluster/work/pghosh/gif1.0/rendered_data/rendered_flame_no_neck_black_bg_chk_texture_flt_teeth_color_norm_map_flame_2020.lmdb'
# rendered_flame_root = '/is/cluster/work/pghosh/gif1.0/rendered_data/rendered_flame_no_neck_black_bg_mean_chk_texture_flt_teeth_color_norm_map_flame_2020.lmdb'
# rendered_flame_root = '/is/cluster/work/pghosh/gif1.0/DECA_inferred/deca_pre_rendered.lmdb'
# rendered_flame_root = '/is/cluster/work/pghosh/gif1.0/DECA_inferred/deca_rendered.lmdb'
rendered_flame_root = '/is/cluster/work/pghosh/gif1.0/DECA_inferred/deca_rendered_with_teture.lmdb'
# rendered_flame_root = '/raid/data/pghosh/face_gan_data/FFHQ/params/rendered_flame_no_neck_rand_bg.lmdb'
# normalization_file_path = '/is/cluster/scratch/partha/gif1.0/FFHQ_dynamicfit_normalization_hawen_parms.npz'
normalization_file_path = None

# flame_param_est = gmm_estimator.DensityEstimator(method_name='GMM', n_components=5)
# flame_param_est.load('/is/cluster/work/pghosh/gif1.0/FLAME_param_generator_dynamic_flm_hawen.pkl')
flame_param_est = None

# import ipdb; ipdb.set_trace()
list_bad_images = np.load('/is/cluster/work/pghosh/gif1.0/DECA_inferred/b_box_stats.npz')['bad_images']
dataset = FFHQ(real_img_root=data_root, rendered_flame_root=rendered_flame_root, params_dir=params_dir,
               generic_transform=generic_transform, pose_cam_from_yao=False,
               rendered_flame_as_condition=True, resolution=256,
               normalization_file_path=normalization_file_path, debug=True, random_crop=True, get_normal_images=True,
               flame_version='DECA', list_bad_images=list_bad_images)

show_img_at_res = 512
with_blur = True

if with_blur:
    blur_module = Blur(channel=None, kernel_len=42, nsig=6, trainable=False)

example_out_dir = '/is/cluster/work/pghosh/gif1.0/DECA_inferred/example_images'

# for i in range(70_000):
#     print(i)
#     dataset.__getitem__(i)

for indx in range(500):
    i = int(random.randint(0, 60_000))
# for bad_img in list_bad_images:
#     i = int(bad_img[:-4])
# for i in range(10):
#     i = 23650
    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5)
    print(f'getting_img{i}')
    img, flm_rndr, flm_lbl, index = dataset.__getitem__(i, bypass_valid_indexing=True)
    # if flm_rndr[0].max() < 0.1:
    #     import ipdb; ipdb.set_trace()
    img = img[None, ...]
    flm_rndr = flm_rndr[0][None, ...]

    if with_blur:
        img_blr = blur_module(img)
        flm_rndr_blr = blur_module(flm_rndr)

        img_blr = fast_image_reshape(img_blr, height_out=show_img_at_res, width_out=show_img_at_res)
        flm_rndr_blr = fast_image_reshape(flm_rndr_blr, height_out=show_img_at_res, width_out=show_img_at_res)

        img_blr = img_blr[0]
        flm_rndr_blr = flm_rndr_blr[0]

    img = fast_image_reshape(img, height_out=show_img_at_res, width_out=show_img_at_res)
    flm_rndr = fast_image_reshape(flm_rndr, height_out=show_img_at_res, width_out=show_img_at_res)

    img = img[0]
    flm_rndr = flm_rndr[0]

    flm_rndr = (flm_rndr.numpy().transpose((1, 2, 0)) + 1)/2
    ax1.imshow((img.numpy().transpose((1, 2, 0)) + 1)/2)
    ax2.imshow(flm_rndr[:, :, :3])
    if with_blur:
        ax4.imshow((img_blr.numpy().transpose((1, 2, 0)) + 1)/2)

    if flm_rndr.shape[-1] > 3:
        ax3.imshow(flm_rndr[:, :, 3:])
        # ax3.imshow(flm_rndr[:, :, 0:3])
        if with_blur:
            flm_rndr_blr = (flm_rndr_blr.numpy().transpose((1, 2, 0)) + 1) / 2
            ax5.imshow(flm_rndr_blr[:, :, 3:])
            # ax4.imshow(flm_rndr_blr[:, :, :3])
    plt.savefig(os.path.join(example_out_dir, str(i)+'.png'))