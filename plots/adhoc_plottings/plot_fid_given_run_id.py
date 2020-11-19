import  sys
sys.path.append('../')
import constants as cnst
import matplotlib.pyplot as plt
import os
import random
# '''Plots training FIDs given a run id for visualization purposes'''

# run_ids = [18, 19]
run_ids = [3]

root_dir = f'{cnst.output_root}sample'

for run_id in run_ids:
    rnd_smpl_files = sorted(os.listdir(os.path.join(root_dir, str(run_id))))
    f_ids = []
    itrns = []
    for fid_at_it in rnd_smpl_files:
        if fid_at_it.endswith('.png'):
            itrn, fid = int(fid_at_it.split('_')[0]), float(fid_at_it.split('_')[3][:-4])
            f_ids.append(fid)
            itrns.append(itrn)

    plt.plot(itrns, f_ids, label=f'run_id: {run_id}')
    plt.ylim((0, 50))
    for i in range(len(f_ids) - 1, 0, -100):
        plt.text(itrns[i], f_ids[i] + 10*random.random()-5, str(f_ids[i]), fontsize=12)
        # import ipdb; ipdb.set_trace()

# import ipdb; ipdb.set_trace()
figure_name = str(run_ids[0])
for i in range(1, len(run_ids)):
    figure_name += '_' + str(run_ids[i])

plt.xlabel('iteration ->')
plt.ylabel('FID ->')
plt.legend()
plt.savefig(figure_name + '.png')