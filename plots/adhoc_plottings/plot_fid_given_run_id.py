import sys
sys.path.append('../../')
import constants as cnst
import matplotlib.pyplot as plt
import os
import random
import numpy as np
# '''Plots training FIDs given a run id for visualization purposes'''

run_ids = [8]
# run_ids = [29]

# root_dir = f'{cnst.output_root}sample'
root_dir = f'/is/cluster/scratch/pghosh/GIF_resources/output_files/sample'

for run_id in run_ids:
    rnd_smpl_files = sorted(os.listdir(os.path.join(root_dir, str(run_id))))
    f_ids = []
    itrns = []
    rnd_smpl_files_with_ckpt = []
    for fid_at_it in rnd_smpl_files:
        if fid_at_it.endswith('.png'):
            itrn, fid = int(fid_at_it.split('_')[0]), float(fid_at_it.split('_')[3][:-4])
            if itrn % 1000 != 0:
                continue
            f_ids.append(fid)
            itrns.append(itrn)
            rnd_smpl_files_with_ckpt.append(fid_at_it)

    plt.plot(itrns, f_ids, label=f'run_id: {run_id}')
    plt.ylim((0, 50))
    for i in range(len(f_ids) - 1, 0, -100):
        plt.text(itrns[i], f_ids[i] + 10*random.random()-5, str(f_ids[i]), fontsize=12)
        # import ipdb; ipdb.set_trace()

    min_fid_i = np.argmin(f_ids)
    plt.text(itrns[min_fid_i], f_ids[min_fid_i] - 5, str(f_ids[min_fid_i]), fontsize=12)
    print(f'Minimum FID for run {run_id} is {f_ids[min_fid_i]} with model {rnd_smpl_files_with_ckpt[min_fid_i]} '
          f'at itr {itrns[min_fid_i]}')

# import ipdb; ipdb.set_trace()
figure_name = str(run_ids[0])
for i in range(1, len(run_ids)):
    figure_name += '_' + str(run_ids[i])

plt.xlabel('iteration ->')
plt.ylabel('FID ->')
plt.legend()
plt.savefig(figure_name + '.png')