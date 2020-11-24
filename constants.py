INDICES = {'SHAPE': (0, 100),
           'EXP': (100, 150),
           'POSE': (150, 156),
           'TRANS': (156, 159),
           'JAW_ROT': (153, 156),
           'GLOBAL_ROT': (150, 153),
           'ROT_JAW_TRANS': (150, 159),
           'ALL': (0, 159)}

DECA_IDX = {'cam': (156, 159),
            'tex': (159, 209),
            'lit': (209, 236)}

def get_idx_list(atrb_name):
    '''atrb_name: tuple of name string or single name string'''
    if atrb_name in INDICES:
        return list(range(*INDICES[atrb_name]))
    else:
        indx_list = []
        for cmp_name in atrb_name:
            indx_list += list(range(*INDICES[cmp_name]))

        return indx_list

############################################# path constants ###########################################################
############################################# Modify according to your needs ###########################################
resources_root = '/is/cluster/scratch/pghosh/GIF_resources'
input_root_dir = f'{resources_root}/input_files'
# DECA and FLAME Resource files
deca_inferred_root = f'{input_root_dir}/DECA_inferred'
deca_data_path = f'{deca_inferred_root}/data/'
flame_resource_path = f'{input_root_dir}/flame_resource/'
flame_texture_space_dat_file = f'{flame_resource_path}texture_data_FLAME_256.npy'

# Geenral parameter and image and artifact generation paths
output_root = f'{resources_root}/output_files/'
save_dir_voca_vid = f'{output_root}voca_video'
voca_flame_seq_file = f'{input_root_dir}sentence36.npz'

ffhq_images_root_dir = '/raid/data/pghosh/face_gan_data/FFHQ/images1024x1024/'



# all_flame_params_file = f'{deca_inferred_root}/deca_flame_params_camera_corrected.npy'
all_flame_params_file = f'{deca_inferred_root}/flame_params_public_texture_model.npy'



true_iamge_lmdb_path = f'{input_root_dir}/FFHQ/multiscale.lmdb'
rendered_flame_root = f'{deca_inferred_root}/deca_rendered_with_public_texture.lmdb'
true_img_stats_dir = f'{deca_inferred_root}/FFHQ/ffhq_fid_stats/'
face_region_mask_file = f'{flame_resource_path}texture_map_256X256_face_only_mask.png'
flm_3_sigmaparams_dir = f'{input_root_dir}/GIF_teaser/data/'
list_deca_failed_iamges = f'{deca_inferred_root}/b_box_stats.npz'

# Placing random images and their rendering side by side
generated_random_image_root_dir = output_root
random_imagesdestination_dir = output_root

# reinference MSE error for evaluation paths

# Amazon Mechanical turk related constants
amt_bucket_base_url = 'https://flameparameterassociation.s3-eu-west-1.amazonaws.com'
five_pt_likert_scale_result_csv_path = f'{input_root_dir}/mturk_results/textured_rend_flm_asso_right_likert_scale.csv'
flame_style_vec_association_result_csv_path = f'{input_root_dir}/mturk_results/Result_flm_asso_10k.csv'

flame_config = {
        # FLAME
        'flame_model_path': f'{flame_resource_path}generic_model.pkl',  # acquire it from FLAME project page
        'flame_lmk_embedding_path': f'{flame_resource_path}landmark_embedding.npy',
        'mesh_file': f'{flame_resource_path}head_template_mesh.obj',

        'tex_space_path': f'{flame_resource_path}FLAME_texture.npz',  # acquire it from FLAME project page
        # 'tex_space_path': f'{flame_resource_path}FLAME_basel_texture.npz',  # acquire it from FLAME project page


        'camera_params': 3,
        'shape_params': 100,
        'expression_params': 50,
        'pose_params': 6,
        'tex_params': 50,
        'use_face_contour': True,

        'cropped_size': 256,
        'batch_size': 1,
        'image_size': 256,
    }
