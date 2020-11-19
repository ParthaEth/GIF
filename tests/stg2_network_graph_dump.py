import torch
import numpy as np
from model.stg2_generator import Generator
from model.stg2_discriminator import Discriminator


if __name__ == "__main__":
    from my_utils.graph_writer import graph_writer
    img_size = 256

    generator = Generator(img_size, 512, 8, channel_multiplier=2)

    # from my_utils.print_model_summary import summary
    # summary(generator, (1, 512))

    graph_writer.draw(generator, 'STG2_Original_Generator.png', (16, 38),
                      [torch.zeros((1, 512), dtype=torch.float32, device='cpu'), ],
                      randomize_noise=False)
    print('Generator modle saved')

    tot_gen_params = 0
    for discrim_params in generator.parameters():
        tot_gen_params += np.prod(discrim_params.shape)
    print(f'generator n_params: {tot_gen_params}')

    discriminator = Discriminator(img_size, channel_multiplier=2)
    graph_writer.draw(discriminator, 'STG2_Original_Discriminator.png', (16, 38),
                      torch.zeros((1, 3, img_size, img_size), dtype=torch.float32, device='cpu'))
    print('Generator modle saved')

    tot_gen_params = 0
    for discrim_params in discriminator.parameters():
        tot_gen_params += np.prod(discrim_params.shape)
    print(f'discriminator n_params: {tot_gen_params}')