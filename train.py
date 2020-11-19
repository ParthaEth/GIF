#!/is/ps2/pghosh/.virtualenvs/gif/bin python

import constants as cnst
import os
import argparse
import math
import numpy as np

from tqdm import tqdm

import torch
from torch import nn, optim
from torch.nn import functional as F

from dataset_loaders import fast_image_reshape, sample_data
from model import stylegan2_common_layers
from model.stg2_generator import StyledGenerator
from model.stg2_discriminator import Discriminator
import torch.backends.cudnn as cudnn
from my_utils import compute_fid, generic_utils
from loss_functions import losses
from datetime import datetime


cudnn.benchmark = True


def train(args, dataset, generator, discriminator_flm, fid_computer, flame_param_est, used_samples,
          step):
    run_avg_rate = 0.999
    interp_loss_run_avg = 0
    pose = None
    if args.embedding_vocab_size != 1:
        true_embeddings = generator.module.get_embddings()

    if args.gen_reg_type.upper() == 'PATH_LEN_REG':
        pl_reg = losses.PathLengthRegularizor()

    fid = np.nan

    if flame_param_est is None:
        fake_flame = None
        fake_indices = None
    else:
        fake_flame = flame_param_est.get_samples(n_samples=50, shuffle=True, normalize=True)
        fake_flame = torch.from_numpy(fake_flame).cuda()
        fake_indices = torch.from_numpy(np.random.randint(0, args.embedding_vocab_size,
                                                          size=fake_flame.shape[0])).cuda()

    viz_saver = generic_utils.VisualizationSaver(gen_i=10, gen_j=5, sampling_flame_labels=fake_flame, dataset=dataset,
                                                 input_indices=fake_indices)

    resolution = 4 * 2 ** step
    resolutions = [4 * 2 ** _step for _step in range(step + 1)]
    loader = sample_data(dataset, args.batch.get(resolution, args.batch_default), resolutions, debug=args.debug)
    data_loader = iter(loader)

    if args.apply_texture_space_interpolation_loss:
        interp_tex_loss = losses.InterpolatedTextureLoss(
            max_images_in_batch=args.batch.get(resolution, args.batch_default))

    # increasing generator learningrate as it falls behind discriminator. After adding more informative loss to
    # discriminator
    generic_utils.adjust_lr(g_optimizer, args.lr.get(resolution, 0.001), args.use_styled_conv_stylegan2)

    pbar = tqdm(range(3_000_000))

    generic_utils.requires_grad(generator, False)

    generic_utils.requires_grad(discriminator_flm, True)
    generic_utils.adjust_lr(d_optimizer_flm, args.lr.get(resolution, 0.001), args.use_styled_conv_stylegan2)

    disc_loss_val = 0
    gen_loss_val = 0
    grad_loss_val = 0

    max_step = int(math.log2(args.max_size)) - 2
    final_progress = False

    for i in pbar:

        generic_utils.requires_grad(discriminator_flm, True)
        discriminator_flm.zero_grad()

        alpha = min(1, 1 / args.phase * (used_samples + 1))

        if (resolution == args.init_size and args.ckpt is None) or final_progress:
            alpha = 1

        # Switching resolution code
        if used_samples > args.phase * 2:
            used_samples = 0
            step += 1

            if step > max_step:
                step = max_step
                final_progress = True

            else:
                alpha = 0

            resolution = 4 * 2 ** step
            time_now = datetime.now()
            print(f'{time_now} : Resolution is : ' + str(resolution))

            resolutions = [4 * 2 ** _step for _step in range(step + 1)]
            loader = sample_data(dataset, args.batch.get(resolution, args.batch_default), resolutions, debug=args.debug)
            data_loader = iter(loader)

            generic_utils.adjust_lr(g_optimizer, args.lr.get(resolution, 0.001), args.use_styled_conv_stylegan2)

        try:
            real_image, flm_rndr, flm_lbls, input_indices = next(data_loader)  # Real image sin different scales
        except (OSError, StopIteration):
            data_loader = iter(loader)
            real_image, flm_rndr, flm_lbls, input_indices = next(data_loader)

        if not args.rendered_flame_as_condition and not args.normal_maps_as_cond:
            flm_rndr = flm_lbls
            dataset.accumulate_batches_of_flm(flm_lbls[0], pose)
        else:
            dataset.accumulate_batches_of_flm(flm_rndr[0], pose)


        real_image = real_image.cuda()
        real_image_list = [fast_image_reshape(real_image, resolutions[-1], resolutions[-1])]

        flm_lbls = flm_lbls[0].cuda()
        flm_rndr = flm_rndr[0].cuda()
        input_indices = input_indices.cuda()

        flm_lbls_with_shuffled_flame = flm_lbls
        flm_rndr_with_shuffled_flame = flm_rndr

        b_size = args.batch.get(resolution, args.batch_default)
        used_samples += b_size

        # real_image_list = [real_img.cuda() for real_img in real_image_list]

        for real_image in real_image_list:  # using grad pen for only the highest res
            real_image.requires_grad = True
        flm_lbls_with_shuffled_flame.requires_grad = not args.texture_space_discrimination

        real_img_condition = flm_rndr_with_shuffled_flame

        real_img_condition.requires_grad = True
        # import ipdb; ipdb.set_trace()
        real_scores_flm, _ = discriminator_flm(real_image_list, condition=real_img_condition, step=step, alpha=alpha)

        real_Dloss_flm = torch.nn.functional.softplus(-real_scores_flm).mean()
        if (i + 1) % 16 == 0:  # To save time do only every 16th iteration. otherwise 17 sec per itr
            # grad_penalty_flm = losses.grad_penalty_loss(real_image_list + [real_img_condition], real_scores_flm,
            #                                             step=None)
            grad_penalty_flm = losses.grad_penalty_loss(real_image_list, real_scores_flm, step=None)
            real_Dloss_flm += grad_penalty_flm.mean()

        if args.rendered_flame_as_condition or args.normal_maps_as_cond:
            gen_in1 = flm_rndr.clone()
        else:
            gen_in1 = flm_lbls.clone()

        # import ipdb; ipdb.set_trace()
        fake_image_list = generator(gen_in1, pose, step=step, alpha=alpha, input_indices=input_indices)
        cond_fake_imgs = gen_in1

        fake_image_list[0] = fake_image_list[0].detach()
        if args.shfld_cond_as_neg_smpl:
            # fake_image_list[0] = torch.cat((fake_image_list[0], real_image_list[0]), dim=0)
            fake_image_list[0] = fake_image_list[0].repeat((2, 1, 1, 1))
            shuffle_indices = generic_utils.get_unique_shuffle_indices(real_img_condition.shape[0])
            cond_discrim_fake_imgs = torch.cat((cond_fake_imgs, real_img_condition[shuffle_indices, :]), dim=0)
        else:
            cond_discrim_fake_imgs = cond_fake_imgs

        fake_scores_flm, _ = discriminator_flm(fake_image_list, condition=cond_discrim_fake_imgs, step=step,
                                               alpha=alpha)

        fake_Dloss = F.softplus(fake_scores_flm).mean()
        (real_Dloss_flm + fake_Dloss).backward()

        disc_loss_val = fake_Dloss + real_Dloss_flm
        disc_loss_val = (disc_loss_val + fake_Dloss).item()

        d_optimizer_flm.step()

        ################################## Training Generator #######################################
        if n_critic >= 1:  # letting generator to be trained more
            if (i + 1) % n_critic == 0:
                gen_itr_count = 1
            else:
                gen_itr_count = 0
        else:
            gen_itr_count = int(1 / n_critic)

        generic_utils.requires_grad(generator, True)

        generic_utils.requires_grad(discriminator_flm, False)

        for gen_trn_itr in range(gen_itr_count):
            generator.zero_grad()

            # import ipdb; ipdb.set_trace()
            fake_image_list = generator(gen_in1, pose, step=step, alpha=alpha, input_indices=input_indices)
            # cond_fake_imgs.detach() is necessary because it will try and propaget gradients and will retain
            # cleared buffers from training on real data and cause double call of backward!
            predict_flm, _ = discriminator_flm(fake_image_list, condition=cond_fake_imgs.detach(), step=step,
                                               alpha=alpha)

            fake_gen_loss = F.softplus(-predict_flm).mean()

            if args.gen_reg_type.upper() == 'PATH_LEN_REG':
                gen_flm_weight = 2
                fake_gen_loss += gen_flm_weight * pl_reg.path_length_reg(generator, step=step, alpha=alpha,
                                                                         input_indices=input_indices)
            elif args.gen_reg_type.upper() == 'DIRECT_GRAD_REG':
                # Changes in flame input should cause as small output change as possible
                gen_flm_weight = 1e-8*8
                fake_gen_loss += gen_flm_weight * losses.grad_penalty_loss(inputs=[gen_in1,],
                                                                           outs=torch.pow(fake_image_list[-1], 2),
                                                                           step=None)

            # embeddign regularizatio loss
            if args.embedding_vocab_size != 1:
                embedding_reg_loss = args.embedding_reg_weight * losses.l2_reg(generator.module.z_to_w)
                fake_gen_loss += embedding_reg_loss

            # interpolation loss. Texture must stay same even when the face is moved with different flame parameters
            if args.apply_texture_space_interpolation_loss:
                # This is the only reason why FLAME labels are necessary in train time
                flm_intrp_batch = flm_lbls[:-1, :159] + \
                                  np.random.uniform(0, 1)*(flm_lbls[1:, :159] - flm_lbls[:-1, :159])
                # During interpolation light and texture code should stay constant. Don in the loss function
                flm_intrp_batch = torch.cat((flm_intrp_batch, flm_lbls[:-1, 159:]), axis=-1)
                # import ipdb; ipdb.set_trace()
                interp_loss = interp_tex_loss.tex_sp_intrp_loss(
                    dataset.un_normalize_flame(flm_intrp_batch), generator,
                    step=step, alpha=alpha,
                    max_ids=args.embedding_vocab_size,
                    normal_maps_as_cond=args.normal_maps_as_cond,
                    use_posed_constant_input=args.use_posed_constant_input,
                    rendered_flame_as_condition=args.rendered_flame_as_condition)
                if args.adaptive_interp_loss:
                    interp_loss *= 0.25*fake_gen_loss.detach()/interp_loss.detach()
                fake_gen_loss += interp_loss

                # import ipdb; ipdb.set_trace()

            fake_gen_loss.backward()
            g_optimizer.step()
            if gen_loss_val is None:
                gen_loss_val = fake_gen_loss.item()
            else:
                gen_loss_val = gen_loss_val * run_avg_rate + fake_gen_loss.item()*(1 - run_avg_rate)

            # decay factor copied from STG2
            generic_utils.accumulate(g_running, generator, decay=0.5 ** (32 / (10 * 1000)))

            generic_utils.requires_grad(generator, False)

        if (i + 1) % 1000 == 0:
            md_chk_pt_name = f'{args.chk_pt_dir}/checkpoint/{str(args.run_id)}/{str(i + 1).zfill(6)}_{alpha}.model'

            chk_pt_dict = {'generator_running': g_running.state_dict(),
                           'generator': generator.state_dict(),
                           'g_optimizer': g_optimizer.state_dict(),
                           'discriminator_flm': discriminator_flm.state_dict(),
                           'd_optimizer_flm': d_optimizer_flm.state_dict()}

            torch.save(chk_pt_dict, md_chk_pt_name)
            np.savez(md_chk_pt_name.replace('.model', '.npz'), step=step, used_sampless=used_sampless, alpha=alpha,
                     resolution=resolution)

        if (i + 1) % 500 == 0:
            # fid_computation
            flame_parmas, input_indices, pose = dataset.get_10k_flame_params()
            # import ipdb; ipdb.set_trace()
            image_tensor = generic_utils.get_images_from_flame_params(flame_parmas, pose, g_running, step=step,
                                                                      alpha=alpha, input_indices=input_indices)
            # import ipdb; ipdb.set_trace()
            fid = fid_computer.get_fid(image_tensor)

        state_msg = (f'Size: {4 * 2 ** step}; G: {gen_loss_val:.3f}; D: {disc_loss_val:.3f};'
                     f' fid: {fid:.0f}')

        if args.embedding_vocab_size != 1:
            state_msg += f', embd_reg_l: {embedding_reg_loss:.3f}; '

        if args.apply_texture_space_interpolation_loss:
            if interp_loss_run_avg is None:
                interp_loss_run_avg = interp_loss.item()
            else:
                interp_loss_run_avg = interp_loss_run_avg*run_avg_rate + interp_loss.item()*(1-run_avg_rate)
            state_msg += f', interp_l: {interp_loss_run_avg:.3f}; '

        pbar.set_description(state_msg)

        if (i + 1) % 500 == 0:
            if flame_param_est is None:
                condition_parmas = torch.from_numpy(flame_parmas[:50]).cuda()
                if pose is not None:
                    pose_for_saving = torch.from_numpy(pose[:50]).cuda()
                else:
                    pose_for_saving = None

                inpt_idxs = torch.from_numpy(input_indices[:50]).cuda()
                viz_saver.set_flame_params(pose_for_saving, condition_parmas, inpt_idxs)
                flame_param_est = 0
            viz_saver.save_samples(i, model=g_running, step=step, alpha=alpha, resolution=resolution, fid=fid,
                                   run_id=args.run_id)


if __name__ == '__main__':
    from configurations import update_config
    # if you get "ModuleNotFoundError: No module named 'past'" do - 'pip install future'
    # Yes I know what you are thinking. Keep calm keep coding

    # Number of critic steps, fractiona vlue is allowed.
    # 1/2 = generator is trained twice for every critic training
    # n_critic = 1/8
    # n_critic = 1/4
    n_critic = 1

    parser = argparse.ArgumentParser(description='Progressive Growing of GANs')
    args, dataset, flame_param_est = update_config(parser)

    if not args.debug:
        fid_computer = compute_fid.FidComputer(database_root_dir=cnst.ffhq_images_root_dir,
                                               true_img_stats_dir=cnst.true_img_stats_dir)
    else:
        fid_computer = None

    os.makedirs(f'{args.chk_pt_dir}/checkpoint/{str(args.run_id)}', exist_ok=True)
    os.makedirs(f'{args.chk_pt_dir}/sample/{str(args.run_id)}', exist_ok=True)

    generator = StyledGenerator(embedding_vocab_size=args.embedding_vocab_size,
                                rendered_flame_ascondition=args.rendered_flame_as_condition,
                                normal_maps_as_cond=args.normal_maps_as_cond,
                                core_tensor_res=args.core_tensor_res,
                                n_mlp=args.nmlp_for_z_to_w,
                                apply_sqrt2_fac_in_eq_lin=args.apply_sqrt_in_eq_linear)

    # from my_utils.print_model_summary import summary
    # summary(generator, (3, 2, 2), device='cpu')

    from my_utils.graph_writer import graph_writer
    graph_writer.draw(generator, f'Style_gan_mdl_run_id{args.run_id}.png', (16, 38),
                      torch.zeros((1, int(3*(int(args.normal_maps_as_cond)+int(args.rendered_flame_as_condition))),
                                   2, 2)))

    generator = nn.DataParallel(generator).cuda()

    discrim_embd_pred_dim = None
    if args.embedding_vocab_size != 1:
        discrim_embd_pred_dim = generator.module.get_embddings().shape[1]

    dscrm_cnd_channels = 3 + int(args.normal_maps_as_cond) * 3 + int(args.rendered_flame_as_condition) * 3

    discriminator_flm = Discriminator(size=args.max_size, num_color_chnls=dscrm_cnd_channels,
                                                              channel_multiplier=2)
    graph_writer.draw(discriminator_flm, f'Style_gan_Discriminator_run_id{args.run_id}.png', (16, 38),
                      [torch.zeros((1, dscrm_cnd_channels, args.max_size, args.max_size))])
    discriminator_flm = nn.DataParallel(discriminator_flm).cuda()

    g_running = nn.DataParallel(StyledGenerator(embedding_vocab_size=args.embedding_vocab_size,
                                                rendered_flame_ascondition=args.rendered_flame_as_condition,
                                                normal_maps_as_cond=args.normal_maps_as_cond,
                                                core_tensor_res=args.core_tensor_res,
                                                n_mlp=args.nmlp_for_z_to_w)).cuda()
    g_running.train(False)

    g_reg_ratio = 4 / (4 + 1)
    d_reg_ratio = 16 / (16 + 1)
    g_optimizer = optim.Adam(generator.module.parameters(), lr=0.002*g_reg_ratio,
                             betas=(0.0, 0.99**g_reg_ratio))

    tot_gen_params = 0
    for gen_params in generator.parameters():
        tot_gen_params += np.prod(gen_params.shape)
    print(f'generator n_params: {tot_gen_params}')

    tot_discrim_params = 0
    for discrim_params in discriminator_flm.parameters():
        tot_discrim_params += np.prod(discrim_params.shape)
    print(f'discriminator n_params: {tot_discrim_params}')


    d_optimizer_flm = optim.Adam(discriminator_flm.parameters(), lr=0.002*d_reg_ratio,
                                 betas=(0.0, 0.99**d_reg_ratio))

    # generic_utils.accumulate(g_running, generator, 0)

    used_sampless = 0
    step = int(math.log2(args.init_size)) - 2

    if args.ckpt is not None:
        ckpt = torch.load(args.ckpt)
        generator.load_state_dict(ckpt['generator'])
        g_optimizer.load_state_dict(ckpt['g_optimizer'])
        discriminator_flm.load_state_dict(ckpt['discriminator_flm'])
        d_optimizer_flm.load_state_dict(ckpt['d_optimizer_flm'])
        g_running.load_state_dict(ckpt['generator_running'])

        other_training_vars = np.load(args.ckpt.replace('.model', '.npz'))
        used_sampless = other_training_vars['used_sampless']
        step = other_training_vars['step']
        print('<=============================== Model state restored! ==============================>')

    train(args, dataset, generator, discriminator_flm, fid_computer, flame_param_est, used_sampless,
          step)
