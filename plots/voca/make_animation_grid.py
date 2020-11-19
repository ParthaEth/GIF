import constants as cnst
import skvideo.io
import os
from PIL import Image
import numpy as np
import tqdm

ids_to_animate = ['selected_ids_1024', 'selected_ids_1069', 'selected_ids_127', 'selected_ids_1427', 'selected_ids_1467']
core_ros1 = ['selected_ids_1468', 'selected_ids_327', 'selected_ids_1472', 'selected_ids_1552', 'selected_ids_1554']
core_row2 = ['selected_ids_355', 'selected_ids_1571', 'mesh', 'selected_ids_1614', 'selected_ids_1642']
core_row3 = ['selected_ids_1683', 'selected_ids_356', 'selected_ids_1904', 'selected_ids_1914', 'selected_ids_1947']
last_row = ['selected_ids_2047', 'selected_ids_238', 'selected_ids_48', 'selected_ids_495', 'selected_ids_663']
ids_to_animate.extend(core_ros1)
ids_to_animate.extend(core_row2)
ids_to_animate.extend(core_row3)
ids_to_animate.extend(last_row)


fps = 30
writer = skvideo.io.FFmpegWriter(os.path.join(cnst.save_dir_voca_vid, f'voca_selected_ids.mp4'),
                                 outputdict={'-r': str(fps)})
padding_in_px = 4
current_frame = np.zeros((5*256+4*padding_in_px, 5*256+4*padding_in_px, 3)).astype(np.uint8)

try:
    for frm_idx in tqdm.tqdm(range(199)):
        # prepare frame
        for i, id in enumerate(ids_to_animate):
            # print(i)
            if id == 'mesh':
                current_file = os.path.join(cnst.save_dir_voca_vid, ids_to_animate[0], f'mesh_textured_{frm_idx}.png')
            else:
                current_file = os.path.join(cnst.save_dir_voca_vid, str(id), f'{frm_idx}.png')
                # import ipdb; ipdb.set_trace()
            current_image = np.array(Image.open(current_file).resize((256, 256)))
            start_row = int(256 * (i // 5) + padding_in_px * (i // 5))
            start_col = int(256 * (i % 5) + padding_in_px * (i % 5))
            current_frame[start_row:int(start_row + 256), start_col:int(start_col + 256), :] = current_image
        writer.writeFrame(current_frame)
finally:
    writer.close()
    print(f'Video saved to: {cnst.save_dir_voca_vid}/voca_selected_ids.mp4')