import os
os.environ['PYTHONHASHSEED'] = '2'
import sys
sys.path.append('../')
import constants as cnst
import tqdm
import numpy as np
from dataset_loaders import FFHQ
from dataset_loaders import sample_data
from torchvision import transforms
from torchvision.utils import make_grid
import skvideo.io
import torch


num_rows = 6
num_cols = 10
duration_in_seconds = 60
fps = 60
frames = int(duration_in_seconds*fps)
speed_px_per_frame = 5

resolution = 256

window_size = (num_rows*resolution, num_cols*resolution)
needed_pixels = window_size[1] + speed_px_per_frame*frames
needed_colums = np.math.ceil(needed_pixels/resolution)

flame_version = 'FLAME_2020_revisited'
texture_pattern = 'MEAN_TEXTURE_WITH_CHKR_BOARD'

expt_name = 'training_data'
save_root = f'{cnst.output_root}sample/groundtruth_data'
save_dir_opn_vid = f'{save_root}/{expt_name}'
os.makedirs(save_dir_opn_vid, exist_ok=True)

normalization_file_path = None
generic_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True), ])


dataset = FFHQ(real_img_root=cnst.true_iamge_lmdb_path, rendered_flame_root=cnst.rendered_flame_root,
               params_dir=cnst.all_flame_params_file, generic_transform=generic_transform, pose_cam_from_yao=False,
               rendered_flame_as_condition=False, resolution=256, normalization_file_path=normalization_file_path,
               debug=False, random_crop=False, get_normal_images=True)
loader = iter(sample_data(dataset, num_rows, None))

frame_img_bigger = None
drop_list = np.array([8, 10, 27, 28, 29, 30, 32, 34])+1

writer = skvideo.io.FFmpegWriter(os.path.join(save_dir_opn_vid, "training_images_video.mp4"),
                                 outputdict={'-r': str(fps)})

batch = 0
try:
    for frm_idx in tqdm.tqdm(range(frames + 5 + len(drop_list))):
        if frame_img_bigger is None or frame_img_bigger.shape[1] <= window_size[1]:
            training_images, rendered_flame, _, _ = next(loader)
            batch += 1
            if batch in drop_list:
                continue
            training_images = make_grid(training_images, nrow=1)
            rendered_flame = make_grid(rendered_flame[0], nrow=1)
            col_image = torch.cat((rendered_flame, training_images), dim=-1).cpu().detach().numpy().transpose((1, 2, 0))
            if frame_img_bigger is None:
                frame_img_bigger = col_image
            else:
                frame_img_bigger = np.concatenate((frame_img_bigger, col_image), axis=1)

        if frame_img_bigger.shape[1] > window_size[1]:
            current_frame = frame_img_bigger[:, :window_size[1]]
            frame_img_bigger = frame_img_bigger[:, speed_px_per_frame:]
            writer.writeFrame(((current_frame + 1)*127.5).astype('uint8'))
finally:
    writer.close()
    print(f'Video saved to: {save_dir_opn_vid}')


