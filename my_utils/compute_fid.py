import sys
sys.path.append('../')

import os
import glob
import numpy as np
from my_utils.pytorch_fid import fid_score
import torch


class FidComputer:
    def __init__(self, database_root_dir, true_img_stats_dir=None):
        self.dims = 2048
        self.true_data_loc = database_root_dir
        self.true_img_stats_dir = true_img_stats_dir
        self.load_inception_model(self.dims)
        self.m_t, self.s_t = None, None
        self.current_resolution = None

    def load_inception_model(self, dims):
        block_idx = fid_score.InceptionV3.BLOCK_INDEX_BY_DIM[dims]
        self.model = fid_score.InceptionV3([block_idx])
        self.model.cuda()
        self.model.eval()

    def compute_true_img_response(self, resolution):
        # dims can take the following vals
        # 64: first max pooling features
        # 192: second max pooling featurs
        # 768: pre-aux classifier features
        # 2048: final average pooling features (this is the default
        true_img_stats_file = os.path.join(self.true_img_stats_dir, f'ffhq_{resolution}X{resolution}_fid_stats.npz')
        print(true_img_stats_file)
        if self.true_img_stats_dir is not None and os.path.exists(true_img_stats_file):
            print('True embedding found loading')
            f = np.load(true_img_stats_file)
            self.m_t, self.s_t = f['mu'][:], f['sigma'][:]
            f.close()
        else:
            # true_iamge_files = self._get_celeba_img_file_path_list()
            # import ipdb; ipdb.set_trace()
            true_iamge_files = glob.glob(os.path.join(self.true_data_loc, '*.png'))[:50000]
            self.m_t, self.s_t = fid_score.calculate_activation_statistics(files=true_iamge_files, model=self.model,
                                                                           batch_size=32, dims=self.dims, cuda=True,
                                                                           resolution=resolution)
            np.savez(true_img_stats_file, mu=self.m_t, sigma=self.s_t)

    def compute_sats_given_img_tensor(self, imag_tensor):
        # imag_tensor: Of shape 10_000 X 3 X 224 X 224 ~ 22.4 GB can stay in CPU RAM
        # Normalized to 0-1. Will perform this normalization anyway
        batch_size = 32
        if type(imag_tensor) is np.ndarray or type(imag_tensor) is torch.Tensor:
            if imag_tensor.dtype == 'float32' or imag_tensor.dtype == torch.float32:
                imag_tensor = imag_tensor - imag_tensor.min()
                imag_tensor /= imag_tensor.max()
            elif imag_tensor.dtype == 'uint8':
                pass
            else:
                raise ValueError('Datatype of Image tensor not undestood: ' + str(imag_tensor.dtype))

        pred_arr = np.zeros((imag_tensor.shape[0], self.dims))

        with torch.no_grad():
            for batch_idx in range(0, imag_tensor.shape[0], batch_size):
                if type(imag_tensor) is np.ndarray:
                    imag_batch = torch.from_numpy(imag_tensor[batch_idx:batch_idx + batch_size].astype('float32')).cuda()
                    if imag_tensor.dtype == 'uint8':
                        imag_batch /= 255
                else:
                    imag_batch = imag_tensor[batch_idx:batch_idx + batch_size].cuda()
                pred_arr[batch_idx:batch_idx + batch_size] = fid_score.compute_activation_batch(self.model, imag_batch)

        mu = np.mean(pred_arr, axis=0)
        sigma = np.cov(pred_arr, rowvar=False)

        return mu, sigma

    def get_fid(self, imag_tensor):
        resolution = imag_tensor.shape[-1]
        if self.m_t is None or self.current_resolution != resolution:
            self.compute_true_img_response(resolution)
            self.current_resolution = resolution

        m2, s2 = self.compute_sats_given_img_tensor(imag_tensor)
        fid_value = fid_score.calculate_frechet_distance(self.m_t, self.s_t, m2, s2)

        return fid_value
