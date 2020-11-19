import sys
sys.path.append('../../')
import glob
import constants as cnst
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True

fontsize = 15


def get_detection_prob_our_model(result_file):
    num_turker = 1
    num_flame_params = 1000
    result_data = pd.read_csv(result_file)
    result_inp_1 = result_data['Input.OPTION1'].tolist()
    result_inp_1_ans = result_data['Answer.example.label1'].to_numpy()
    correct_detection_count = 0
    # import ipdb;ipdb.set_trace()
    for i in range(len(result_inp_1)):
        if (result_inp_1[i].find('mdl1') >= 0 and result_inp_1_ans[i]) or \
           (result_inp_1[i].find('mdl2') >= 0 and not result_inp_1_ans[i]):
            correct_detection_count += 1

    return correct_detection_count/(num_turker*num_flame_params)


full_mdl = ['full_model']
ablated = ['vector_cond', 'flm_rndr_tex_interp', 'norm_mp_tex_interp', 'norm_map_rend_flm_no_tex_interp']
# ablated = ['flm_rndr_tex_interp', 'norm_mp_tex_interp', 'norm_map_rend_flm_no_tex_interp']
prefix = ''

detection_probs = []
for model_original in full_mdl:
    for model_abl in ablated:
        print(f'{prefix}{model_original}_VS_{model_abl}*.csv')
        result_file = glob.glob(
            f'{cnst.output_root}sample/inter_model_comparison/results/'
            f'{prefix}{model_original}_VS_{model_abl}*.csv')[0]
        detect_prob_our_model = get_detection_prob_our_model(result_file)
        detection_probs.append(detect_prob_our_model)

x = list(range(len(ablated)))
print((x, detection_probs))
plt.plot(x, detection_probs, '^', markersize=12)
ax = plt.gca()
# ax.set_xlabel(r'Random ID', fontsize=fontsize)
plt.xticks(x, [f'Our_Model_VS_{model_abl}'.replace('_', '\_') for model_abl in ablated], rotation=10)
ax.set_ylabel(r' Detection accuracy $\rightarrow$', fontsize=fontsize)
ax.tick_params(axis='both', which='both', labelsize=fontsize)
ax.plot(x, [0.5, ] * len(x))
ax.legend(['Turker performance', 'Random Chance'], fontsize=fontsize)
# plt.show()
plt.savefig('moel_comparisons.png', bbox_inches='tight')