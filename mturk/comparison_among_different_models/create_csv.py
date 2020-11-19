import sys
sys.path.append('../../')
import constants as cnst
import pandas
import numpy as np
import os
import glob

full_mdl = ['full_model']
ablated = ['vector_cond', 'flm_rndr_tex_interp', 'norm_mp_tex_interp', 'norm_map_rend_flm_no_tex_interp']

base_img_path = f'{cnst.output_root}/model_comparison/sample/inter_model_comparison/'


for model_original in full_mdl:
    mdl_1_path = os.path.join(base_img_path, model_original)
    list_images_mdl1 = glob.glob(os.path.join(mdl_1_path, 'mdl1_*'))
    # import ipdb; ipdb.set_trace()
    for model_abl in ablated:

        list_1 = []
        list_2 = []
        mesh_files = []
        np.random.seed(2)

        for mdl_1_img in list_images_mdl1:
            mdl_1_img = os.path.basename(mdl_1_img)
            original_model_goes_left = np.random.randint(2)

            mdl_1_img_full_url = cnst.amt_bucket_base_url + model_original + '/' + mdl_1_img
            mesh_files.append(mdl_1_img_full_url.replace('mdl1_', 'mesh'))
            mdl_2_full_url = mdl_1_img_full_url.replace(model_original, model_abl).replace('mdl1_', 'mdl2_')
            if original_model_goes_left:
                list_1.append(mdl_1_img_full_url)
                list_2.append(mdl_2_full_url)
            else:
                list_1.append(mdl_2_full_url)
                list_2.append(mdl_1_img_full_url)

            # import ipdb; ipdb.set_trace()

        df = pandas.DataFrame(data={"GT": mesh_files, "OPTION1": list_1, "OPTION2":list_2})
        csv_path = f'{cnst.output_root}sample/inter_model_comparison/input_csvs'
        df.to_csv(os.path.join(csv_path, f'{model_original}_VS_{model_abl}.csv'), sep=',', index=False)