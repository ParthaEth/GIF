import os
from torchvision import transforms
from dataset_loaders import FFHQ
import constants as cnst
import numpy as np


def parse_args(parser):
    parser.add_argument('--run_id', default=0, type=int, help='Running process id')
    parser.add_argument('--lr', default=0.002, type=float, help='learning rate')
    parser.add_argument('--sched', default=True, help='use lr scheduling')
    parser.add_argument('--init_size', default=8, type=int, help='initial image size')
    parser.add_argument('--max_size', default=256, type=int, help='max image size')
    parser.add_argument('--ckpt', default=None, type=str, help='load from previous checkpoints')
    parser.add_argument('--debug', default=False, type=bool,
                        help='Lowers batchsize and nuber of flame params loaded for quick debug')
    try:
        savae_dir_default = os.environ['SM_MODEL_DIR']
    except KeyError as e:
        # Kept for sage maker id the env. var is not set just go with empty string
        savae_dir_default = '.'
    parser.add_argument('--chk_pt_dir', type=str, default=savae_dir_default)

    parser.add_argument(
        '--no_from_rgb_activate',
        action='store_true',
        help='use activate in from_rgb (original implementation)',
    )
    args = parser.parse_args()
    return args


def update_config(parser):
    args = parse_args(parser)

    generic_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),])

    if args.run_id == 3:
        args.conditional_discrim = True
        args.flame_dims = 159
        args.embedding_vocab_size = -1
        args.embedding_reg_weight = 0
        args.gen_reg_type = 'None'
        args.use_also_img_only_discrim = False
        args.use_inst_norm = True
        args.use_posed_constant_input = False
        args.use_styled_conv_stylegan2 = True
        args.shfld_cond_as_neg_smpl = False
        args.core_tensor_res = 4
        args.init_size = 256
        args.max_size = 256
        args.nmlp_for_z_to_w = 8
        args.adaptive_interp_loss = False

        args.normal_maps_as_cond = True
        args.add_low_res_shortcut = False
        args.rendered_flame_as_condition = False
        args.apply_texture_space_interpolation_loss = True

        params_dir = cnst.all_flame_params_file
        data_root = cnst.true_iamge_lmdb_path
        rendered_flame_root = cnst.rendered_flame_root
        normalization_file_path = None

        args.apply_sqrt_in_eq_linear = False
        flame_param_est = None

        # import ipdb; ipdb.set_trace()
        list_bad_images = np.load(cnst.list_deca_failed_iamges)['bad_images']
        dataset = FFHQ(real_img_root=data_root, rendered_flame_root=rendered_flame_root, params_dir=params_dir,
                       generic_transform=generic_transform, pose_cam_from_yao=False,
                       rendered_flame_as_condition=args.rendered_flame_as_condition, resolution=256,
                       normalization_file_path=normalization_file_path, debug=args.debug, random_crop=False,
                       get_normal_images=args.normal_maps_as_cond, flame_version='DECA',
                       list_bad_images=list_bad_images)
        args.phase = 600_000/5

    elif args.run_id == 7:
        args.conditional_discrim = True
        args.flame_dims = 159
        args.embedding_vocab_size = -1
        args.embedding_reg_weight = 0
        args.gen_reg_type = 'None'
        args.use_also_img_only_discrim = False
        args.use_inst_norm = True
        args.use_posed_constant_input = False
        args.use_styled_conv_stylegan2 = True
        args.shfld_cond_as_neg_smpl = False
        args.core_tensor_res = 4
        args.init_size = 256
        args.max_size = 256
        args.nmlp_for_z_to_w = 8

        args.normal_maps_as_cond = False
        args.add_low_res_shortcut = False
        args.rendered_flame_as_condition = True
        args.apply_texture_space_interpolation_loss = False

        params_dir = cnst.all_flame_params_file
        data_root = cnst.true_iamge_lmdb_path
        rendered_flame_root = cnst.rendered_flame_root
        normalization_file_path = None

        args.apply_sqrt_in_eq_linear = False
        flame_param_est = None

        # import ipdb; ipdb.set_trace()
        list_bad_images = np.load(cnst.list_deca_failed_iamges)['bad_images']
        dataset = FFHQ(real_img_root=data_root, rendered_flame_root=rendered_flame_root, params_dir=params_dir,
                       generic_transform=generic_transform, pose_cam_from_yao=False,
                       rendered_flame_as_condition=args.rendered_flame_as_condition, resolution=256,
                       normalization_file_path=normalization_file_path, debug=args.debug, random_crop=False,
                       get_normal_images=args.normal_maps_as_cond, flame_version='DECA',
                       list_bad_images=list_bad_images)
        args.phase = 600_000/5

    elif args.run_id == 8:
        args.conditional_discrim = True
        args.flame_dims = 159
        args.embedding_vocab_size = -1
        args.embedding_reg_weight = 0
        args.gen_reg_type = 'None'
        args.use_also_img_only_discrim = False
        args.use_inst_norm = True
        args.use_posed_constant_input = False
        args.use_styled_conv_stylegan2 = True
        args.shfld_cond_as_neg_smpl = False
        args.core_tensor_res = 4
        args.init_size = 256
        args.max_size = 256
        args.nmlp_for_z_to_w = 8

        args.normal_maps_as_cond = True
        args.add_low_res_shortcut = False
        args.rendered_flame_as_condition = True
        args.apply_texture_space_interpolation_loss = False

        params_dir = cnst.all_flame_params_file
        data_root = cnst.true_iamge_lmdb_path
        rendered_flame_root = cnst.rendered_flame_root
        normalization_file_path = None

        args.apply_sqrt_in_eq_linear = False
        flame_param_est = None

        # import ipdb; ipdb.set_trace()
        list_bad_images = np.load(cnst.list_deca_failed_iamges)['bad_images']
        dataset = FFHQ(real_img_root=data_root, rendered_flame_root=rendered_flame_root, params_dir=params_dir,
                       generic_transform=generic_transform, pose_cam_from_yao=False,
                       rendered_flame_as_condition=args.rendered_flame_as_condition, resolution=256,
                       normalization_file_path=normalization_file_path, debug=args.debug, random_crop=False,
                       get_normal_images=args.normal_maps_as_cond, flame_version='DECA',
                       list_bad_images=list_bad_images)
        args.phase = 600_000/5

    elif args.run_id == 29:
        args.conditional_discrim = True
        args.flame_dims = 159
        args.embedding_vocab_size = -1
        args.embedding_reg_weight = 0
        args.gen_reg_type = 'None'
        args.texture_space_discrimination = False
        args.use_also_img_only_discrim = False
        args.use_inst_norm = True
        args.use_posed_constant_input = False
        args.use_styled_conv_stylegan2 = True
        args.shfld_cond_as_neg_smpl = False
        args.core_tensor_res = 4
        args.init_size = 256
        args.max_size = 256
        args.nmlp_for_z_to_w = 8
        args.adaptive_interp_loss = False

        args.normal_maps_as_cond = True
        args.add_low_res_shortcut = False
        args.rendered_flame_as_condition = True
        args.apply_texture_space_interpolation_loss = True

        params_dir = cnst.all_flame_params_file
        data_root = cnst.true_iamge_lmdb_path
        rendered_flame_root = cnst.rendered_flame_root
        normalization_file_path = None

        args.apply_sqrt_in_eq_linear = False
        flame_param_est = None

        # import ipdb; ipdb.set_trace()
        list_bad_images = np.load(cnst.list_deca_failed_iamges)['bad_images']
        dataset = FFHQ(real_img_root=data_root, rendered_flame_root=rendered_flame_root, params_dir=params_dir,
                       generic_transform=generic_transform, pose_cam_from_yao=False,
                       rendered_flame_as_condition=args.rendered_flame_as_condition, resolution=256,
                       normalization_file_path=normalization_file_path, debug=args.debug, random_crop=False,
                       get_normal_images=args.normal_maps_as_cond, flame_version='DECA',
                       list_bad_images=list_bad_images)
        args.phase = 600_000/5

    else:
        raise ValueError(f'Unknown configuration! {args.run_id}')

    if args.sched and not args.debug:
        # args.lr = {128: 0.0015, 256: 0.003126094126388248, 512: 0.003, 1024: 0.003}
        args.lr = {128: 0.0, 256: 0.0, 512: 0.0, 1024: 0.0}
        args.batch = {4: 512, 8: 256, 16: 128, 32: 64, 64: 32, 128: 32, 256: 16, 512: 16, 1024: 16}  # can take bigger batches!
        # args.batch = {128: 20, 256: 20, 512: 16}  # can take bigger batches!

    else:
        args.lr = {}
        args.batch = {}

    args.gen_sample = {512: (8, 4), 1024: (4, 2)}

    if args.debug:
        args.batch_default = 4
    else:
        args.batch_default = 16

    if args.embedding_vocab_size != 1:
        args.embedding_vocab_size = len(dataset)

    if args.use_styled_conv_stylegan2:
        print('<<<<<<<<<< Running Style GAN 2 >>>>>>>>>>>>>>>>>>>>>>>')

    return args, dataset, flame_param_est
