import sys
sys.path.append('../')
from model import StyledGenerator, Discriminator
import torch
import numpy as np

generator = StyledGenerator(flame_dim=159,
                            all_stage_discrim=False,
                            embedding_vocab_size=70_000,
                            rendered_flame_ascondition=False,
                            inst_norm=True,
                            normal_maps_as_cond=True,
                            core_tensor_res=4,
                            use_styled_conv_stylegan2=True,
                            n_mlp=8)


# set all weights to 1s
mdl_state = generator.state_dict()
torch.manual_seed(2)
# tot_params = 0
# for name in mdl_state:
#     if name.find('z_to_w') >= 0 or name.find('generator') >= 0 and name.find('embd') < 0 and \
#         name.find('to_rgb.8') < 0 and name.find('to_rgb.7') < 0 and name.find('progression.8') < 0 \
#             and name.find('progression.7') < 0:
#         print(name)
#         mdl_state[name] = mdl_state[name] * 0 + torch.randn(mdl_state[name].shape)
#         tot_params += np.prod(mdl_state[name].shape)
#     else:
#         mdl_state[name] = mdl_state[name] * 0 + 6e-3
#
# print(f'Total set params are: {tot_params}')

tot_params = 0
for name in mdl_state:
    if name.find('z_to_w') >= 0:
        print(name)
        mdl_state[name] = mdl_state[name] * 0 + torch.randn(mdl_state[name].shape)
        tot_params += np.prod(mdl_state[name].shape)
    else:
        mdl_state[name] = mdl_state[name] * 0 + 6e-3

print(f'Total set params are: {tot_params} \n\n\n\n\n')

tot_params = 0
for i in range(7):
    for name in mdl_state:
        if name.find(f'progression.{i}.') >= 0:
            mdl_state[name] = mdl_state[name] * 0 + torch.randn(mdl_state[name].shape)
            tot_params += np.prod(mdl_state[name].shape)
            print(f'{name} : {mdl_state[name].shape}; params this layer: {np.prod(mdl_state[name].shape)}')
    # else:
    #     mdl_state[name] = mdl_state[name] * 0 + 6e-3

print(f'Total set params are: {tot_params} \n\n\n\n\n')

tot_params = 0
for i in range(7):
    for name in mdl_state:
        if name.find(f'to_rgb.{i}') >= 0:
            mdl_state[name] = mdl_state[name] * 0 + torch.randn(mdl_state[name].shape)
            tot_params += np.prod(mdl_state[name].shape)
            print(f'{name} : {mdl_state[name].shape}; params this layer: {np.prod(mdl_state[name].shape)}')
        # else:
        #     mdl_state[name] = mdl_state[name] * 0 + 6e-3

print(f'Total set params are: {tot_params} \n\n\n\n\n')

generator.load_state_dict(mdl_state)

input_indices = torch.zeros((1, ), dtype=torch.long)
flm_rndr = torch.zeros((1, 3, 4, 4))

torch.manual_seed(2)
forward_pass_gen = generator(flm_rndr, pose=None, step=6, alpha=1, input_indices=input_indices)
print(forward_pass_gen)
print(forward_pass_gen[0].shape)

# for param in generator.parameters():
#     print(param)