import numpy as np


light_code = []
fl_param_dict = np.load('/is/cluster/work/pghosh/gif1.0/DECA_inferred/deca_flame_params_camera_corrected.npy',
                        allow_pickle=True).item()
for i, key in enumerate(fl_param_dict):
    flame_param = fl_param_dict[key]
    light_code.append(flame_param['lit'].flatten())

light_code = np.array(light_code)
mean = np.mean(light_code, axis=0)
std = np.std(light_code, axis=0)

print(f'mean : {mean} \n std: {std}')
print(f'most variation in cmp {np.argmax(std)}')
