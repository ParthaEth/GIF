import torch
import torchvision
import dataset_loaders
from my_utils.visualize_flame_overlay import OverLayViz
import numpy as np
try:
    from my_utils.flm_dynamic_fit_overlay import camera_dynamic, camera_ringnetpp
except ModuleNotFoundError:
    camera_ringnetpp = None

import imageio
import os
import tqdm


def get_unique_shuffle_indices(n):
    """
    :param n: an integer n
    :return: a shuffled index numpy array that displaces all the indices from their original position
    """
    indices = np.arange(0, n)
    np.random.shuffle(indices)
    while (indices == np.arange(0, n)).any():
        np.random.shuffle(indices)

    return indices


def get_images_from_flame_params(flame_params, pose, model, step, alpha, input_indices):
    batch_size = 16
    model.eval()
    img_tensor = []
    flame_params = torch.from_numpy(flame_params)
    if pose is not None:
        pose = torch.from_numpy(pose)
    input_indices = torch.from_numpy(input_indices)
    pose_this_batch = None
    with torch.no_grad():
        for b_id in range(0, flame_params.shape[0], batch_size):
            # import ipdb; ipdb.set_trace()
            flame_param_this_batch = flame_params[b_id:b_id + batch_size].cuda()
            input_indices_this_batch = input_indices[b_id:b_id + batch_size].cuda()
            if pose is not None:
                pose_this_batch = pose[b_id:b_id + batch_size].cuda()
            # import ipdb; ipdb.set_trace()
            img_batch = model(flame_param_this_batch, pose_this_batch, step=step, alpha=alpha,
                              input_indices=input_indices_this_batch)[-1]
            img_tensor.append(torch.clamp(img_batch.cpu(), -1, 1))

    img_tensor = torch.cat(img_tensor, dim=0)
    return img_tensor


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def accumulate(model1, model2, decay=0.999):
    '''
    Implementation of running Averages
        input:
            model1: nn.Module
            model2: nn.Module

        model1_params = model1_params*decay + (1-decay)*model2_params
    '''
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(1 - decay, par2[k].data)


def adjust_lr(optimizer, lr, style_gn2):
    if not style_gn2:
        for group in optimizer.param_groups:
            mult = group.get('mult', 1)
            group['lr'] = lr * mult


class VisualizationSaver():
    def __init__(self, gen_i, gen_j, sampling_flame_labels, dataset, input_indices, overlay_mesh=False):
        self.gen_i = gen_i
        self.gen_j = gen_j
        self.sampling_flame_labels = sampling_flame_labels
        self.overlay_mesh = overlay_mesh
        self.overlay_visualizer = OverLayViz()
        self.dataset = dataset
        self.input_indices = input_indices
        self.cam_t = np.array([0., 0., 2.5])

    def set_flame_params(self, pose, sampling_flame_labels, input_indices):
        self.pose = pose
        self.sampling_flame_labels = sampling_flame_labels
        self.input_indices = input_indices

    def save_samples(self, i, model, step, alpha, resolution, fid, run_id):
        images = []
        # camera_params = camera_dynamic((resolution, resolution), self.cam_t)
        flength = 5000
        cam_t = np.array([0., 0., 0])
        camera_params = camera_ringnetpp((512, 512), trans=cam_t, focal=flength)

        with torch.no_grad():
            for img_idx in range(self.gen_i):
                flame_param_this_batch = self.sampling_flame_labels[img_idx * self.gen_j:(img_idx + 1) * self.gen_j]
                if self.pose is not None:
                    pose_this_batch = self.pose[img_idx * self.gen_j:(img_idx + 1) * self.gen_j]
                else:
                    pose_this_batch = None
                idx_this_batch = self.input_indices[img_idx * self.gen_j:(img_idx + 1) * self.gen_j]
                img_tensor = model(flame_param_this_batch.clone(), pose_this_batch, step=step, alpha=alpha,
                                   input_indices=idx_this_batch)[-1]

                img_tensor = self.overlay_visualizer.range_normalize_images(
                    dataset_loaders.fast_image_reshape(img_tensor, height_out=256, width_out=256,
                                                       non_diff_allowed=True))

                images.append(img_tensor.data.cpu())

        torchvision.utils.save_image(
            torch.cat(images, 0),
            f'sample/{str(run_id)}/{str(i + 1).zfill(6)}_res{resolution}x{resolution}_fid_{fid:.2f}.png',
            nrow=self.gen_i,
            normalize=True,
            range=(0, 1))


def save_set_of_images(path, prefix, images, show_prog_bar=False, name_list=None):
    os.makedirs(path, exist_ok=True)
    if len(images.shape) == 4:  # Color channels present
        if images.shape[-1] == 1:
            images = images[:, :, :, 0]
        elif images.shape[-1] != 1 and images.shape[1] == 3:
            images = images.transpose((0, 2, 3, 1))

    if images.dtype =='float32':
        if np.min(images) < 0 or np.max(images) > 1:
            print('Warning! Not in the range of 0 - 1')

        images = (np.clip(images, 0, 1)*255).astype('uint8')


    if show_prog_bar:
        progbar = tqdm.tqdm(range(images.shape[0]))
        progbar.set_description('Saving images')
        for i in progbar:
            if name_list is None:
                cur_name = i
            else:
                cur_name = name_list[i]
            imageio.imwrite(os.path.join(path, prefix + f'{cur_name}.png'), images[i])
    else:
        for i, img in enumerate(images):
            if name_list is None:
                cur_name = i
            else:
                cur_name = name_list[i]
            imageio.imwrite(os.path.join(path, prefix + f'{cur_name}.png'), img)
