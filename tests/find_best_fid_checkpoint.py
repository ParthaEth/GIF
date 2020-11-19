import os
import numpy as np
run_id = 3

root_dir = '/is/cluster/work/pghosh/gif1.0'
checkpoint_dir = os.path.join(root_dir, 'checkpoint', str(run_id))
smpl_dir = os.path.join(root_dir, 'sample', str(run_id))

smpl_files = os.listdir(smpl_dir)
fids = []
smpl_png_files = []
for smpl_file in smpl_files:
    if smpl_file.endswith('.png'):
        fids.append(float(smpl_file.split('_')[-1][:-4]))
        smpl_png_files.append(smpl_file)

sorted_fid_idx = np.argsort(fids)
# print(sorted_fid_idx)
sorted_smpl_files = np.array(smpl_png_files)[sorted_fid_idx]
# print(sorted_smpl_files)

checkpoint_files = os.listdir(checkpoint_dir)
ckpt_itrs = []
ck_pt_postfix = []
for chk_pt_file in checkpoint_files:
    if chk_pt_file.endswith('.model'):
        ckpt_itrs.append(chk_pt_file.split('_')[0])
        ck_pt_postfix.append('_' + '_'.join(chk_pt_file.split('_')[1:]))

for smpl_file in sorted_smpl_files:
    # import ipdb; ipdb.set_trace()
    iteration_idx = smpl_file.split('_')[0]
    try:
        chk_pt_idx_found = ckpt_itrs.index(iteration_idx)
    except ValueError:
        continue
    if chk_pt_idx_found >= 0:
        print(f'Best FID checkpoit = {ckpt_itrs[chk_pt_idx_found] + ck_pt_postfix[chk_pt_idx_found]}, its fid: {smpl_file}')
        # break