import torchvision.transforms.functional as F
import os
import torch
import glob
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import tqdm
import pickle
import numpy as np
import random

from io import BytesIO
import lmdb

from constants import INDICES

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
  with open(path, 'rb') as f:
    img = Image.open(f)
    img = img.convert('RGB')

    return img


def fast_image_reshape(in_img_batch, height_out, width_out, non_diff_allowed=False, mode='bicubic'):
    resize_img = torch.nn.functional.interpolate(in_img_batch, size=(width_out, height_out), mode=mode)

    if non_diff_allowed:
        min_pix = in_img_batch.min().item()
        max_pix = in_img_batch.max().item()
        resize_img = resize_img.clamp(min=min_pix, max=max_pix)

    return resize_img


def shuffle_flame_params(original_flame_params, fake_flame_params):
    batch_size = original_flame_params.shape[0]
    new_flame_batch = original_flame_params.repeat((5, 1))

    shape_idx = INDICES['SHAPE']
    exp_idx = INDICES['EXP']
    cam_idx = INDICES['CAM']
    pose_idx = INDICES['POSE']

    # Swap shape
    new_flame_batch[batch_size:batch_size*2, shape_idx[0]:shape_idx[1]] = fake_flame_params[:, shape_idx[0]:shape_idx[1]]

    # Swap expression
    new_flame_batch[batch_size*2:batch_size*3, exp_idx[0]:exp_idx[1]] = fake_flame_params[:, exp_idx[0]:exp_idx[1]]

    # Swap pose
    new_flame_batch[batch_size*3:batch_size*4, pose_idx[0]:pose_idx[1]] = fake_flame_params[:, pose_idx[0]:pose_idx[1]]

    # Swap camera
    new_flame_batch[batch_size*4:batch_size*5, cam_idx[0]:cam_idx[1]] = fake_flame_params[:, cam_idx[0]:cam_idx[1]]

    return new_flame_batch


def same_padding_crop(img, normalized_crop):
    """
    Not in place same padding crop however destroyes img
    normalized_crop: tuple of float32/ float64 crop size if the image height and width is 1. i.e. 1 being max
    rows or cols
    """
    img_new = img.clone()
    row_crop = int(normalized_crop[0] * img.shape[1])
    col_crop = int(normalized_crop[1] * img.shape[2])
    rows, cols = img.shape[1:]

    if row_crop != 0:
        if row_crop > 0:  # shift up
            img_new[:, :rows-row_crop, :] = img[:, row_crop:, :]
            img_new[:, rows-row_crop:, :] = img[:, rows-row_crop:rows-row_crop+1, :]
        else:  # shift down
            row_crop = -row_crop
            img_new[:, row_crop:, :] = img[:, :rows-row_crop, :]
            img_new[:, :row_crop, :] = img[:, 0:1, :]

    img = img_new.clone()
    if col_crop != 0:
        if col_crop > 0:  # shift left
            img_new[:, :, :cols-col_crop] = img[:, :, col_crop:]
            img_new[:, :, cols-col_crop:] = img[:, :, cols-col_crop:cols-col_crop+1]
        else:  # shift Right
            col_crop = -col_crop
            img_new[:, :, col_crop:] = img[:, :, :cols-col_crop]
            img_new[:, :, :col_crop] = img[:, :, 0:1]

    return img_new


class FFHQ(Dataset):
    '''
        Flame parameters for FFHQ are located in
        /ps/project/face2d3d/faceHQ_100K/faceHQ_100K_fitting/flame_dynamic/params
    '''
    def __init__(self, real_img_root, rendered_flame_root, params_dir, generic_transform, normalization_file_path,
                 rendered_flame_as_condition, resolution=256, debug=False, pose_cam_from_yao=False,
                 generate_flame_only=False, random_crop=False, get_normal_images=False,
                 flame_version='FLAME_2020_revisited', camera=None, apply_random_h_flip=False, list_bad_images = []):
        self.generic_transfor = generic_transform
        self.scaling_transforms = []
        self.real_imgae_root = real_img_root
        self.params_dir = params_dir
        self.pose_cam_from_yao = pose_cam_from_yao
        self.all_flm_parmas = None
        self.generate_flame_only = generate_flame_only
        self.rendered_flame_as_condition = rendered_flame_as_condition
        self.random_crop = random_crop
        self.crop_max_in_px = 2
        self.get_normal_images = get_normal_images
        self.flame_version = flame_version
        self.camera = camera
        self.apply_random_h_flip = apply_random_h_flip
        self.list_bad_images = list_bad_images
        np.random.seed(2)
        torch.manual_seed(2)

        self.ffhq_params = self.collect_params(flame_version, debug)
        self.valid_ids = self.get_valid_ids()

        if flame_version == 'FLAME_2020_revisited':
            self.rend_flm_res = 512
            if self.camera is None:
                flength = 5000
                self.camera = {'f': [flength, flength], 'c': [self.rend_flm_res/2, self.rend_flm_res/2]}
        else:
            self.rend_flm_res = 256

        if pose_cam_from_yao:
            self.flm_parmas = np.load(os.path.join(self.params_dir, 'FFHQ_ringnet_params.npz'), allow_pickle=True)
            self.all_flm_parmas = np.hstack((self.flm_parmas['shape'], self.flm_parmas['exp'], self.flm_parmas['pose'],
                                             self.flm_parmas['camera']))
            # there are missing files! So weare collecting the contiguous index of the current file name to access the
            # right flame param index in the numpy file
            self.paramfile_id_to_index = {int(k1.split('.')[0]): k2 for k1, k2 in zip(sorted(self.ffhq_params.keys()),
                                                                   list(range(0, len(self.ffhq_params))))}

        if pose_cam_from_yao:
            self.flame_mean = np.mean(self.all_flm_parmas, axis=0)
            self.flame_std = np.std(self.all_flm_parmas, axis=0)
        elif flame_version == 'FLAME_2020_revisited':
            assert normalization_file_path is None
            self.flame_mean = 0
            self.flame_std = 1
        elif flame_version == 'DECA':
            assert normalization_file_path is None
            self.flame_mean = 0
            self.flame_std = 1
        else:
            normalization_data = np.load(normalization_file_path)
            self.flame_mean = normalization_data['mean'][0]
            self.flame_std = normalization_data['std'][0]

        if not generate_flame_only:
            self.env_real_images = lmdb.open(
                real_img_root,
                max_readers=32,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,)
            if not self.env_real_images:
                raise IOError('Cannot open lmdb dataset', real_img_root)
            else:
                with self.env_real_images.begin(write=False) as txn:
                    self.length = int(txn.get('length'.encode('utf-8')).decode('utf-8'))

        if self.rendered_flame_as_condition or self.get_normal_images:
            self.env_rendered_images = lmdb.open(
                rendered_flame_root,
                max_readers=32,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,)
            if not self.env_rendered_images:
                raise IOError('Cannot open lmdb dataset', real_img_root)


        self.resolution = resolution

        self.img = np.random.uniform(-1, 1, (3, 1024, 1024)).astype('float32')
        self.flm_lbl = [np.random.uniform(-1, 1, 119).astype('float32')]
        self.flm_col_idx = 0
        self.flm_10k_params = None
        self.pose_10_k = None

    def un_normalize_flame(self, flame_batch):
        if self.flame_mean != 0 and self.flame_std != 1:
            return flame_batch * torch.from_numpy(self.flame_std).cuda() + torch.from_numpy(self.flame_mean).cuda()
        else:
            return flame_batch

    def set_resolution(self, resolution):
        self.resolution = resolution

    def accumulate_batches_of_flm(self, flm_batch, pose):
        if self.flm_col_idx < 10_000:
            flm_batch = flm_batch.cpu().numpy().astype('float32')
            if self.flm_10k_params is None:
                self.flm_10k_params = np.zeros((10_000, ) + flm_batch.shape[1:], dtype='float32')
            if self.pose_10_k is None and pose is not None:
                pose = pose.cpu().numpy().astype('float32')
                self.pose_10_k = np.zeros((10_000,) + pose.shape[1:], dtype='float32')

            max_acumulatable = min(flm_batch.shape[0], 10_000 - self.flm_col_idx)
            self.flm_10k_params[self.flm_col_idx:self.flm_col_idx + max_acumulatable, :] = \
                flm_batch[:max_acumulatable, :]
            if pose is not None:
                self.pose_10_k[self.flm_col_idx:self.flm_col_idx + max_acumulatable, :] = pose[:max_acumulatable, :]
            self.flm_col_idx += max_acumulatable

    def get_10k_flame_params(self):
        if self.pose_cam_from_yao:
            # TODO(Partha): Not always 'all_flm_parmas' are avialable. accumulate over batches!
            flm_parms_10k = self.all_flm_parmas[:10_000]
            flm_parms_10k = (flm_parms_10k.astype('float32') - self.flame_mean) / self.flame_std
            return flm_parms_10k, np.arange(10_000), self.pose_10_k
        else:
            return self.flm_10k_params, np.arange(10_000), self.pose_10_k

    def apply_transforms_to_images(self, img):
        if self.generic_transfor:  # Be careful not to destroy the flame param association
            img = self.generic_transfor(img)

        if self.scaling_transforms:
            img = [transform(F.to_pil_image(img)).float() for transform in self.scaling_transforms]
        else:
            img = img.float()

        return img

    def __getitem__(self, index, bypass_valid_indexing=False):
        if bypass_valid_indexing:
            index_valid = index
        else:
            index_valid = self.valid_ids[index]
            curren_file = str(index_valid).zfill(5) + '.npy'
            # import ipdb; ipdb.set_trace()
            while curren_file in self.list_bad_images:
                # print(f'oops!: {curren_file} is bad')
                index_valid = self.valid_ids[np.random.randint(0, len(self.valid_ids))]
                curren_file = str(index_valid).zfill(5) + '.npy'

        if self.generate_flame_only:
            img = 0
        else:
            with self.env_real_images.begin(write=False) as txn:
                key = f'{self.resolution}-{str(index_valid).zfill(5)}'.encode('utf-8')
                img_bytes = txn.get(key)

            buffer = BytesIO(img_bytes)
            img = Image.open(buffer)
            img = self.apply_transforms_to_images(img)

        if self.rendered_flame_as_condition or self.get_normal_images:
            with self.env_rendered_images.begin(write=False) as txn:
                if self.rendered_flame_as_condition:
                    key = f'{self.rend_flm_res}-{str(index_valid).zfill(5)}'.encode('utf-8')
                    img_bytes = txn.get(key)
                    # import ipdb; ipdb.set_trace()
                if self.get_normal_images:
                    normal_img_key = f'norm_map_{self.rend_flm_res}-{str(index_valid).zfill(5)}'.encode('utf-8')
                    normal_img_bytes = txn.get(normal_img_key)

            if self.rendered_flame_as_condition:
                buffer = BytesIO(img_bytes)
                flm_rndr = Image.open(buffer)
                if flm_rndr.size[0] != self.resolution:
                    flm_rndr = flm_rndr.resize((self.resolution, self.resolution))

            if self.get_normal_images:
                norm_img_buffer = BytesIO(normal_img_bytes)
                normal_img = Image.open(norm_img_buffer)
                # print(self.resolution)
                if normal_img.size[0] != self.resolution:
                    normal_img = normal_img.resize((self.resolution, self.resolution))

            if self.get_normal_images and not self.rendered_flame_as_condition:
                flm_rndr = normal_img

            if self.rendered_flame_as_condition and self.get_normal_images:
                flm_rndr = [torch.cat((self.apply_transforms_to_images(flm_rndr),
                                       self.apply_transforms_to_images(normal_img)), dim=0)]
            else:
                flm_rndr = [self.apply_transforms_to_images(flm_rndr)]
        else:
            flm_rndr = [0]

        if self.pose_cam_from_yao:
            flame_param = self.all_flm_parmas[self.paramfile_id_to_index[index_valid]]
        else:
            flame_param = self.ffhq_params[str(index_valid).zfill(5)+'.pkl']
            if self.flame_version == 'old':
                flame_param = np.hstack((flame_param['betas'][:100], flame_param['betas'][300:350],
                                         flame_param['pose'][0:3], flame_param['pose'][6:9], flame_param['trans']))
            elif self.flame_version == 'FLAME_2020_revisited':
                flame_param = np.hstack(
                    (flame_param['shape'], flame_param['exp'], flame_param['pose'], flame_param['cam']))
                tz = self.camera['f'][0] / (self.camera['c'][0] * flame_param[:, 156:157])
                flame_param[:, 156:159] = np.concatenate((flame_param[:, 157:], tz), axis=1)
                flame_param = flame_param[0]
            elif self.flame_version == 'DECA':
                # import ipdb; ipdb.set_trace()
                flame_param = np.hstack(
                    (flame_param['shape'], flame_param['exp'], flame_param['pose'], flame_param['cam'],
                     flame_param['tex'], flame_param['lit'].flatten()))

        flm_lbl = [(flame_param.astype('float32') - self.flame_mean)/self.flame_std]

        if self.random_crop:
            flm_lbl[0] *= 0
            crop_256 = np.random.randint(-self.crop_max_in_px, self.crop_max_in_px, 2)
            # if index == 11:
            #     import ipdb; ipdb.set_trace()
            img = same_padding_crop(img, crop_256/256.0)
            flm_rndr[0] = same_padding_crop(flm_rndr[0], crop_256/256.0)

        if self.apply_random_h_flip:
            flm_lbl[0] *= 0  # TODO(Partha): If you figure out how to get FLAME param for hflip it might not be made None
            flm_lbl[0] -= 9999
            if random.random() < 0.5:
                img = torch.flip(img, [-1])
                flm_rndr[0] = torch.flip(flm_rndr[0], [-1])

        return img, flm_rndr, flm_lbl, index

    def collect_params(self, flame_version, debug):
        '''
            Puts all the FLAME parameters of FFHQ images in a dict and returns the dict
            Structure of output dict:
            params_dict = {'filename1.pkl': {'trans': np.array(size[3]), 'betas': np.array(size[400]),
                                           'rotation': np.array(size[4]), 'pose': np.array(size[15]),}
                           'filename2.pkl':  {'trans': np.array(size[3]), 'betas': np.array(size[400]),
                                           'rotation': np.array(size[4]), 'pose': np.array(size[15]),}
                           'filename3.pkl':  {'trans': np.array(size[3]), 'betas': np.array(size[400]),
                                           'rotation': np.array(size[4]), 'pose': np.array(size[15]),}
                           ...
            }
        '''
        print('Collating FFHQ parameters')
        if flame_version == 'old':
            file_ext_to_look_for = '*.pkl'
        elif flame_version == 'FLAME_2020_revisited' or flame_version == 'DECA':
            file_ext_to_look_for = '*.npy'
        else:
            raise ValueError(f'flame version {flame_version} not understood')

        if os.path.isdir(self.params_dir):
            params_dict = {}
            params_files = glob.glob(os.path.join(self.params_dir, file_ext_to_look_for))
            params_files_celebA = glob.glob(os.path.join(self.params_dir, 'imgHQ' + file_ext_to_look_for))
            param_files_FFHQ = sorted(list(set(params_files) - set(params_files_celebA)))
            for i, f in enumerate(tqdm.tqdm(param_files_FFHQ)):
                key = os.path.basename(f)[:-3] + 'pkl'
                param = self.load_flame_param(f)
                params_dict[key] = param
                if i >= 2000 and debug:
                    break
            # np.save('flame_dynamic_2020.npy', params_dict)
            # exit(0)
        elif self.params_dir.endswith('.npy'):
            # import ipdb; ipdb.set_trace()
            params_dict = np.load(self.params_dir, allow_pickle=True).item()

        print('Collating FFHQ parameters, done!')
        return params_dict

    def load_flame_param(self, param_path):
        if param_path.endswith('.npy'):
            params = np.load(param_path, allow_pickle=True)
        else:
            with open(param_path, 'rb') as param_file:
                params = pickle.load(param_file, encoding='latin1')
        return params

    def get_valid_ids(self):
        valid_ids = []
        for id in self.ffhq_params.keys():
            valid_ids.append(int(id.split('.')[0]))
        return valid_ids

    def __len__(self):
        return len(self.valid_ids)


def sample_data(dataset, batch_size, image_sizes, debug=False):
    if debug:
        n_workers = 0
    else:
        n_workers = 16
    loader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=n_workers, drop_last=True,
                        pin_memory=True)
    return loader
