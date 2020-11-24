import sys
sys.path.append('../')
import constants as cnst
import torch

from my_utils.photometric_optimization import gif_helper
from my_utils.photometric_optimization import util


class OverLayViz:
    def __init__(self):
        self._random_seed = 2
        torch.manual_seed(self._random_seed)
        config_obj = util.dict2obj(cnst.flame_config)
        self.rendering_helper = gif_helper.render_utils(config_obj)

    def get_rendered_mesh(self, flame_params, camera_params, cull_backfaces=False, constant_albedo=None):
        if cull_backfaces:
            raise ValueError('Back face culling option not avialable make a feature request to '
                             'photometric_optimization repository')
        shape, expression, pose, lightcode, texcode = flame_params

        textured_images, normal_images = \
            self.rendering_helper.render_tex_and_normal(shapecode=shape, expcode=expression,
                                                        posecode=pose, texcode=texcode,
                                                        lightcode=lightcode, cam=camera_params,
                                                        constant_albedo=constant_albedo)
        # import ipdb; ipdb.set_trace()
        textured_images = torch.floor(textured_images.clamp(0, 255))/255.0
        # textured_images = textured_images.clamp(0, 1)
        normal_images = torch.floor(normal_images.clamp(0, 1) * 255)/255.0

        return normal_images, None, None, None, textured_images.type(torch.float)


    @staticmethod
    def range_normalize_images(in_img):
        max_pix = in_img.max()
        min_pix = in_img.min()
        return (in_img - min_pix)/(max_pix - min_pix)
