import sys
sys.path.append('../../../')
import constants as cnst
import skvideo.io
from PIL import Image
import os
import numpy as np
import tqdm

fps = 30
# no '/' at the end of save_dir
save_dir = f'{cnst.output_root}/sample/29/teaser_figure/interpolations'
writer = skvideo.io.FFmpegWriter(os.path.join(save_dir, f'my_vid.mp4'),
                                 outputdict={'-r': str(fps)})
                                 
# images_root_dir = './images'
images_root_dir = f'{cnst.output_root}sample/29/teaser_figure/interpolations'
person1 = 5
tex1 = 6
person2 = 7
tex2 = 7
blank_size = 60
images_matrix = [
                [
                    [['mesh',''], 'shape', [1, 0,  0]],
                    [[person1, tex1], 'shape', [1, 0,  0]],
                    [[person2, tex2], 'shape', [1, 0, 0]],
                    [['mesh',''], 'shape', [0, 1,  0]],
                    [[person1, tex1], 'shape', [0, 1,  0]],
                    [[person2, tex2], 'shape', [0, 1, 0]],
                    # [['mesh',''], 'shape', [0, 0,  1]],
                    # [1, 'shape', [0, 0,  1]],
                    # [2, 'shape', [0, 0, 1]],
                ],
                [
                    [['mesh',''], 'exp', [1, 0,  0]],
                    [[person1, tex1], 'exp', [1, 0,  0]],
                    [[person2, tex2], 'exp', [1, 0, 0]],
                    [['mesh',''], 'exp', [0, 1,  0]],
                    [[person1, tex1], 'exp', [0, 1,  0]],
                    [[person2, tex2], 'exp', [0, 1, 0]],
                    # [['mesh',''], 'exp', [0, 0,  1]],
                    # [1, 'exp', [0, 0,  1]],
                    # [2, 'exp', [0, 0, 1]],
                ],
                [
                    [['mesh',''], 'pose', ['comp4', '-pi_8', '+pi_8']],
                    [[person1, tex1], 'pose', ['comp4', '-pi_8', '+pi_8']],
                    [[person2, tex2], 'pose', ['comp4', '-pi_8', '+pi_8']],
                    [['mesh',''], 'pose', ['comp6', '0', '+pi_12']],
                    [[person1, tex1], 'pose', ['comp6', '0', '+pi_12']],
                    [[person2, tex2], 'pose', ['comp6', '0', '+pi_12']],

                ],
                [
                    [['mesh',''], 'light', ['-3', '+3']],
                    [[person1, tex1], 'light', ['-3', '+3']],
                    [[person2, tex2], 'light', ['-3', '+3']],
                    [['mesh',''], 'albedo', ['-3', '+3']],
                    [[person1, tex1], 'albedo', ['-3', '+3']],
                    [[person2, tex2], 'albedo', ['-3', '+3']],

                ],



            ]
img_matrix_rows, img_matrix_columns = len(images_matrix), len(images_matrix[0])
print(img_matrix_rows, img_matrix_columns)
patch_size = (256, 256)

total_frames = 64
cols = 4
rows = 4
r_idx, c_idx = 0, 0
# ./images/00_-3_00_shape_VS_00_+3_00_shape/images_id_20_tex_6/meshes/mesh_textured_29.png

for frame in tqdm.tqdm(range(0, total_frames)):
    img_frame = np.zeros(((256+blank_size)*img_matrix_rows, 256*img_matrix_columns, 3))
    r_idx = 0
    for matrix_row in images_matrix:
        c_idx = 0
        for matrix_col in matrix_row:

            identity, texture_val = matrix_col[0]
            feature = matrix_col[1]

            if feature == 'shape' or feature == 'exp':
                p0, p1, p2 = matrix_col[2]
                if p0:
                    feature_folder = f'-3_00_00_{feature}_VS_+3_00_00_{feature}'
                elif p1:
                    feature_folder = f'00_-3_00_{feature}_VS_00_+3_00_{feature}'
                elif p2:
                    feature_folder = f'00_00_-3_{feature}_VS_00_00_+3_{feature}'

            elif feature == 'pose':
                p0, s, e = matrix_col[2]
                feature_folder = f'{p0}_{s}_pose_VS_{p0}_{e}_pose/'

            elif feature == 'albedo' or feature == 'light':
                s, e = matrix_col[2]
                feature_folder = f'{s}_{feature}_VS_{e}_{feature}/'

            if identity == 'mesh':
                img_path = os.path.join(images_root_dir, feature_folder, 'images_id_20_tex_6', 'meshes',
                                        f'mesh_textured_{frame}.png')
                img = np.array(Image.open(img_path).resize((256, 256)))
            else:
                img_path = os.path.join(images_root_dir, feature_folder, f'images_id_20_tex_{texture_val}', 'images', f'{identity}',
                                        f'{identity}_{frame}.png')
                img = np.array(Image.open(img_path))

            if not os.path.exists(img_path):
                print(f'***************** Image does not exist {img_path} *****************')
            else:
                #import ipdb; ipdb.set_trace()
                #print(r_idx, c_idx)
                img_frame[(256+blank_size)*r_idx:(256+blank_size)*r_idx+256, 256*c_idx:256*(c_idx+1), :] = img

                #print(img_path)

            c_idx += 1
        r_idx += 1

    writer.writeFrame(img_frame)
    img_frame = Image.fromarray(img_frame.astype(np.uint8))
    if img_frame.mode != 'RGB':
        img_frame = img_frame.convert('RGB')

    img_frame.save(f'{save_dir}/{frame}.png')

        #print(img.shape)
writer.close()
                                 

