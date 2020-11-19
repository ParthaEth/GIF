import sys
sys.path.append('../')
from my_utils.DECA.decalib.DECA import DECA, get_config
from my_utils.DECA.decalib.nets.FLAME import FLAMETex
from my_utils.DECA.decalib import util
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


flame_datapath='/is/cluster/work/pghosh/gif1.0/DECA_inferred/data_deca_flame/'
config = get_config(flame_datapath)
# flametex = FLAMETex(config).to('cuda')
deca = DECA(datapath=flame_datapath, device='cuda')

face_region_only_indices = np.load('/is/cluster/work/pghosh/gif1.0/flame_resource/face_mask.npy', allow_pickle=True)
# import ipdb; ipdb.set_trace()
models = ['full_model', 'flm_rndr_tex_interp', 'norm_map_rend_flm_no_tex_interp', 'vector_cond',
          'norm_mp_tex_interp']
# models = ['full_model', 'vector_cond']
for model in models:
    print('\n\n' + model + ': ')
    errors = []
    percent_errors = []
    for code_type in ['shape', 'exp', 'pose']:
        print(f'{code_type}: ')
        for img_id in range(1024):
            my_flame_params = f'/is/cluster/scratch/partha/gif_eval_data_toheiven/random_samples_q_eval_{model}/params.npy'
            dea_inf_path = f'/is/cluster/scratch/partha/gif_eval_data_toheiven/deca_results/_iter648000/' \
                           f'random_samples_q_eval_{model}/images/{img_id}.npy'

            my_flame_params = np.load(my_flame_params, allow_pickle=True).item()
            deca_params = np.load(dea_inf_path, allow_pickle=True)


            shapecode = []
            expcode = []
            posecode = []
            cam = []
            texcode = []
            lightcode = []

            shapecode.append(torch.tensor(deca_params['shape'])[None, ...].cuda())
            shapecode.append(torch.tensor(my_flame_params['shape'][img_id])[None, ...].cuda())

            expcode.append(torch.tensor(deca_params['exp'])[None, ...].cuda())
            expcode.append(torch.tensor(my_flame_params['exp'][img_id])[None, ...].cuda())

            posecode.append(torch.tensor(deca_params['pose'])[None, ...].cuda())
            posecode.append(torch.tensor(my_flame_params['pose'][img_id])[None, ...].cuda())

            cam.append(torch.tensor(my_flame_params['cam'][img_id])[None, ...].cuda())
            cam.append(torch.tensor(my_flame_params['cam'][img_id])[None, ...].cuda())

            texcode.append(torch.tensor(deca_params['tex'])[None, ...].cuda())
            texcode.append(torch.tensor(deca_params['tex'])[None, ...].cuda())
            # texcode.append(torch.tensor(my_flame_params['tex'][0])[None, ...].cuda())

            lightcode.append(torch.tensor(deca_params['lit'])[None, ...].cuda())
            lightcode.append(torch.tensor(deca_params['lit'])[None, ...].cuda())
            # lightcode.append(torch.tensor(my_flame_params['lit'][0])[None, ...].cuda())

            shapecode = torch.cat(shapecode, dim=0)
            expcode = torch.cat(expcode, dim=0)
            posecode = torch.cat(posecode, dim=0)
            cam = torch.cat(cam, dim=0)
            texcode = torch.cat(texcode, dim=0)
            lightcode = torch.cat(lightcode, dim=0)


            # render
            if code_type == 'shape':
                verts, landmarks2d, landmarks3d = deca.flame(shape_params=shapecode, expression_params=expcode*0,
                                                             pose_params=posecode*0)
            elif code_type == 'exp':
                verts, landmarks2d, landmarks3d = deca.flame(shape_params=shapecode * 0, expression_params=expcode,
                                                             pose_params=posecode)
            elif code_type == 'pose':
                verts, landmarks2d, landmarks3d = deca.flame(shape_params=shapecode * 0, expression_params=expcode * 0,
                                                             pose_params=posecode)

            landmarks2d_projected = util.batch_orth_proj(landmarks2d, cam)
            landmarks2d_projected[:, :, 1:] *= -1
            trans_verts = util.batch_orth_proj(verts, cam)
            trans_verts[:, :, 1:] = -trans_verts[:, :, 1:]

            # # print(landmarks2d_projected.shape)
            # landmarks2d_projected = landmarks2d_projected.cpu().detach().numpy()
            # landmarks2d_projected = landmarks2d_projected * 128 + 128
            # original_image = mpimg.imread('/is/cluster/scratch/partha/gif_eval_data_toheiven/random_samples_q_eval_full_model/images/0.png')
            # # f, axes = plt.subplots(1, 1, sharey=True)
            # plt.imshow(original_image)
            # plt.plot(landmarks2d_projected[0, :, 0], landmarks2d_projected[0, :, 1], 'r*')
            #
            # # axes[1].imshow(original_image)
            # plt.plot(landmarks2d_projected[1, :, 0], landmarks2d_projected[1, :, 1], 'g*')
            # plt.savefig(f'./deca_re_inf_test/{img_id}.png')

            # print(f'norm_diff projected landmarks : {np.linalg.norm(landmarks2d_projected[0, :, 0:2] - landmarks2d_projected[1, :, 0:2])}')
            # print(f'original norm : {np.linalg.norm(landmarks2d_projected[0, :, 0:2])}')
            verts_face_region = verts[:, face_region_only_indices, :].detach().cpu().numpy()

            vertex_dist = np.linalg.norm(verts_face_region[0, :, :], axis=-1)
            mm_error = np.linalg.norm(verts_face_region[0, :, :] - verts_face_region[1, :, :], axis=-1)
            errors.append(np.mean(mm_error))
            percent_errors.append(np.mean(mm_error / vertex_dist))

        # import ipdb; ipdb.set_trace()
        print(f'mean mse error vertices (mm): {np.mean(errors) * 1000}')
        print(f'mean frac error vertices : {np.mean(percent_errors)}')