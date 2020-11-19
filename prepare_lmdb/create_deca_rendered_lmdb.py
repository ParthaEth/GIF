import sys
sys.path.append('../')
import constants as cnst
import tqdm
import numpy as np
import os
import torch
from my_utils.DECA.decalib.DECA import DECA, get_config
from my_utils.DECA.decalib import util
import cv2
import lmdb
from io import BytesIO
from PIL import Image
import time
from my_utils.DECA.decalib.nets.FLAME import FLAMETex


num_validation_images = -1
num_files = 70_000
# num_files = 70
batch_size = 32
# num_files = 4
resolution = 256
flame_param_dict = np.load(cnst.all_flame_params_file, allow_pickle=True).item()
param_files = flame_param_dict.keys()

flame_datapath='../data/'
config = get_config(flame_datapath)
flametex = FLAMETex(config).to('cuda')
deca = DECA(datapath=flame_datapath, device='cuda')

with lmdb.open(cnst.rendered_flame_root, map_size=1024 ** 4, readahead=False) as env:
    with env.begin(write=True) as transaction:
        total = 0
        for batch_id in tqdm.tqdm(range(num_files//batch_size + 1)):
            shapecode = []
            expcode = []
            posecode = []
            cam = []
            texcode = []
            lightcode = []
            for item_id in range(batch_size):
                i = batch_id*batch_size + item_id
                start_time = time.time()
                param_file_name = str(i).zfill(5) + '.pkl'
                try:
                    params = flame_param_dict[param_file_name]
                except KeyError:
                    continue

                shapecode.append(torch.tensor(params['shape'])[None, ...].cuda())
                expcode.append(torch.tensor(params['exp'])[None, ...].cuda())
                posecode.append(torch.tensor(params['pose'])[None, ...].cuda())
                cam.append(torch.tensor(params['cam'])[None, ...].cuda())
                texcode.append(torch.tensor(params['tex'])[None, ...].cuda())
                lightcode.append(torch.tensor(params['lit'])[None, ...].cuda())

            shapecode = torch.cat(shapecode, dim=0)
            expcode = torch.cat(expcode, dim=0)
            posecode = torch.cat(posecode, dim=0)
            cam = torch.cat(cam, dim=0)
            texcode = torch.cat(texcode, dim=0)
            lightcode = torch.cat(lightcode, dim=0)

            # render
            verts, landmarks2d, landmarks3d = deca.flame(shape_params=shapecode, expression_params=expcode,
                                                         pose_params=posecode)
            landmarks2d_projected = util.batch_orth_proj(landmarks2d, cam)
            landmarks2d_projected[:, :, 1:] *= -1
            trans_verts = util.batch_orth_proj(verts, cam)
            trans_verts[:, :, 1:] = -trans_verts[:, :, 1:]

            albedos = flametex(texcode)
            rendering_results = deca.render(verts, trans_verts, albedos, lights=lightcode, light_type='point')
            textured_images, normals = rendering_results['images'], rendering_results['normals']
            normal_images = deca.render.render_normal(trans_verts, normals)

            count = 0
            for item_id in range(batch_size):
                i = batch_id * batch_size + item_id
                if not str(i).zfill(5) + '.pkl' in param_files:
                    continue
                textured_image = cv2.resize(util.tensor2image(textured_images[count]), (resolution, resolution))
                textured_image = Image.fromarray(textured_image)

                normal_image = cv2.resize(util.tensor2image(normal_images[count]), (resolution, resolution))
                # print(i)
                if normal_image.shape[0] != resolution:
                    raise ValueError('None image or something weird happenned!')
                normal_image = Image.fromarray(normal_image)

                # Flame rendering
                key = f'{resolution}-{str(i).zfill(5)}'.encode('utf-8')
                buffer = BytesIO()
                textured_image.save(buffer, format='png', quality=100)
                val = buffer.getvalue()
                transaction.put(key, val)

                key_norm_map = f'norm_map_{resolution}-{str(i).zfill(5)}'.encode('utf-8')
                # print(key_norm_map)
                normal_img_buffer = BytesIO()
                normal_image.save(normal_img_buffer, format='png', quality=100)
                norm_val = normal_img_buffer.getvalue()
                transaction.put(key_norm_map, norm_val)
                total += 1
                count += 1

        transaction.put('length'.encode('utf-8'), str(total).encode('utf-8'))