import sys
sys.path.append('../')
import constants as cnst
import tqdm
import numpy as np
import torch
import lmdb
from io import BytesIO
from PIL import Image
import time
from my_utils.visualize_flame_overlay import OverLayViz
# import matplotlib.pyplot as plt


overlay_viz = OverLayViz()
num_validation_images = -1
num_files = 70_000
batch_size = 32
resolution = cnst.flame_config['image_size']
flame_param_dict = np.load(cnst.all_flame_params_file, allow_pickle=True).item()
param_files = flame_param_dict.keys()

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
            normal_images, _, _, _, textured_images = \
                overlay_viz.get_rendered_mesh((shapecode, expcode, posecode, lightcode, texcode), cam)

            count = 0
            for item_id in range(batch_size):
                i = batch_id * batch_size + item_id
                if not str(i).zfill(5) + '.pkl' in param_files:
                    continue
                textured_image = (textured_images[count].detach().cpu().numpy()*255).astype('uint8').transpose((1, 2, 0))
                textured_image = Image.fromarray(textured_image)

                normal_image = (normal_images[count].detach().cpu().numpy()*255).astype('uint8').transpose((1, 2, 0))
                normal_image = Image.fromarray(normal_image)

                # # Just for inspection
                # fig, (ax1, ax2,) = plt.subplots(1, 2)
                # ax1.imshow(textured_image)
                # ax2.imshow(normal_image)
                # plt.savefig(f'{batch_id}_{item_id}.png')

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