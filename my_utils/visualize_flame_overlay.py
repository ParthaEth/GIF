import sys
sys.path.append('../')
import constants as cnst
import torch


try:
    from my_utils.DECA.decalib.DECA import DECA, get_config
    from my_utils.DECA.decalib.nets.FLAME import FLAMETex
    from my_utils.DECA.decalib import util
except ImportError:
    print('Could not import deca texture stealing will be unavailable')


class OverLayViz:
    def __init__(self, add_random_noise_to_background=False, full_neck=False, background_img=None,
                 texture_pattern_name='CHKR_BRD_FLT_TEETH', inside_mouth_faces=True,
                 flame_version='FLAME_2020_revisited', image_size=224, rendering_method='DECA',
                 deca_conf=None):
        self._random_seed = 2
        torch.manual_seed(self._random_seed)
        self.render = None
        self.batch_size = -1
        self.flame_decoder = None
        self.add_random_noise_to_background = add_random_noise_to_background
        self.full_neck = full_neck
        self.background_img = background_img
        self.texture_pattern_name = texture_pattern_name
        self.inside_mouth_faces = inside_mouth_faces
        self.flame_version = flame_version
        self.image_size = image_size
        self.rendering_method = 'DECA'
        if rendering_method == 'DECA':
            if deca_conf is None:
                config = get_config(cnst.deca_data_path)
            else:
                config = deca_conf
            self.flametex = FLAMETex(config).to('cuda')
            self.deca = DECA(datapath=cnst.deca_data_path, device='cuda', config=deca_conf)

    def get_rendered_mesh(self, flame_params, camera_params, cull_backfaces=False, grey_texture=False):
        shape, expression, pose, lightcode, texcode = flame_params
        # render
        # import ipdb; ipdb.set_trace()
        verts, landmarks2d, landmarks3d = self.deca.flame(shape_params=shape, expression_params=expression,
                                                          pose_params=pose)
        landmarks2d_projected = util.batch_orth_proj(landmarks2d, camera_params)
        landmarks2d_projected[:, :, 1:] *= -1
        trans_verts = util.batch_orth_proj(verts, camera_params)
        trans_verts[:, :, 1:] = -trans_verts[:, :, 1:]

        if grey_texture:
            albedos = torch.tensor([0.6, 0.6, 0.6], dtype=torch.float32)[None, ..., None, None].cuda()
            albedos = albedos.repeat(texcode.shape[0], 1, 256, 256)
            # albedos[-4:] = fast_image_reshape(right_albedos[-4:], height_out=512, width_out=512)
        else:
            albedos = self.flametex(texcode)
        rendering_results = self.deca.render(verts, trans_verts, albedos, lights=lightcode, light_type='point',
                                             cull_backfaces=cull_backfaces)
        textured_images, normals, alpha_img = rendering_results['images'], rendering_results['normals'],\
                                              rendering_results['alpha_images']
        normal_images = self.deca.render.render_normal(trans_verts, normals)

        return normal_images, None, alpha_img, landmarks2d_projected, textured_images.type(torch.float)


    @staticmethod
    def range_normalize_images(in_img):
        max_pix = in_img.max()
        min_pix = in_img.min()
        return (in_img - min_pix)/(max_pix - min_pix)

    def get_overlayed_image(self, alpha, original_image, shape_img, alpha_images, pos_mask):
        # original_image = np.array(original_image.resize((224, 224))).astype('float32') / 255
        original_image = self.range_normalize_images(in_img=original_image[:])

        shape_image = (alpha_images * shape_img * pos_mask)
        overlay_transparent = (1.0 - alpha) * original_image + alpha * shape_image
        overlay_img = ((1 - alpha_images) * original_image + alpha_images * overlay_transparent)
        return overlay_img
