import sys
sys.path.append('../')
import constants as cnst
import os
import torch
import tqdm
import numpy as np
import constants


SHAPE = [0, 1, 2]
EXP = [50, 51, 52]
POSE = [150, 151, 152, 153, 154, 155]


def centre_using_nearest(flame_seq, flame_dataset, one_translation_for_whole_seq=True):
    shape_weigth = 0
    pose_weight = 0.7

    if one_translation_for_whole_seq:
        dist = np.linalg.norm(flame_dataset[:, 150:156] - flame_seq[0, 150:156], axis=-1)
        min_arg = np.argmin(dist)
        flame_seq[:, 156:] = flame_dataset[min_arg, 156:]
    else:
        for i in range(len(flame_seq)):
            shape_dist = np.linalg.norm(flame_dataset[:, SHAPE] - flame_seq[i, SHAPE], axis=-1)
            pose_dist = np.linalg.norm(flame_dataset[:, POSE] - flame_seq[i, POSE], axis=-1)
            dist = shape_weigth*shape_dist + pose_weight*pose_dist
            min_arg = np.argmin(dist)
            flame_seq[i, 156:] = flame_dataset[min_arg, 156:]

    return flame_seq


def position_to_given_location(deca_flame_decoder, flame_batch):
    # import ipdb;
    # ipdb.set_trace()
    shape, expression, pose = (flame_batch[:, 0:100], flame_batch[:, 100:150], flame_batch[:, 150:156])
    verts, _, _ = deca_flame_decoder(shape_params=shape, expression_params=expression, pose_params=pose)

    for i in range(verts.shape[0]):
        e_1_3d = verts[i, 4051, :]
        e_2_3d = verts[i, 4597, :]

        eye_3d_mat = torch.zeros(size=(3, 4)).to(flame_batch.device)
        eye_3d_mat[1, 0] = eye_3d_mat[1, 1] = eye_3d_mat[2, 2] = eye_3d_mat[2, 3] = 1
        eye_3d_mat[0, 0] = e_1_3d[0]
        eye_3d_mat[0, 1] = e_2_3d[0]
        eye_3d_mat[0, 2] = e_1_3d[1]
        eye_3d_mat[0, 3] = e_2_3d[1]

        normalized_image_desired_positions_x1_x2_y1_y2 = \
            torch.tensor([-0.2419, 0.2441, 0.0501-0.1, 0.0509-0.1]).to(flame_batch.device)

        s, s_b_x, s_b_y = torch.matmul(normalized_image_desired_positions_x1_x2_y1_y2, torch.pinverse(eye_3d_mat))
        b_x = s_b_x/s
        b_y = s_b_y/s
        s = -s

        # import ipdb;
        # ipdb.set_trace()
        flame_batch[i, 156] = s
        flame_batch[i, 157] = b_x
        flame_batch[i, 158] = b_y

    return flame_batch


def translate_to_center_eye(flame_decoder, flame_params, original_flame):
    shape, expression, pose, translation = (flame_params[:, 0:100,], flame_params[:, 100:150], flame_params[:, 150:156],
                                            flame_params[:, 156:159])
    verts, _ = flame_decoder(shape_params=shape, expression_params=expression, pose_params=pose,
                                       translation=translation*0)

    if original_flame is not None:
        shape_orig, expression_orig, pose_orig, translation_orig = (original_flame[:, 0:100,],
                                                                    original_flame[:, 100:150],
                                                                    original_flame[:, 150:156],
                                                                    original_flame[:, 156:159])
        verts_orig, _ = flame_decoder(shape_params=shape_orig, expression_params=expression_orig,
                                           pose_params=pose_orig, translation=translation_orig)

        desired_cntr_of_the_eyes = verts_orig[:, 3666, :]
    else:
        desired_cntr_of_the_eyes = torch.from_numpy(np.array([4.32830852e-02, -47.60086733e-03,  2.41298008e+00])
                                                    .astype('float32')).to(flame_params.device)
        # desired_cntr_of_the_eyes = torch.from_numpy(np.array([2.2427477e-03, -1.8124590e-02, 2.5114515e+00])
        #                                             .astype('float32')).to(flame_params.device)

    current_translation = verts[:, 3666, :]

    required_translation = desired_cntr_of_the_eyes - current_translation
    return torch.cat((shape, expression, pose, required_translation), dim=1)


class RegressorNNSkipPart(torch.nn.Module):
    def __init__(self, neurons, regularization, num_layers_per_block, activation_type):
        super().__init__()

        layers = []
        for layer_idx in range(num_layers_per_block):
            layers.append(torch.nn.Linear(neurons, neurons, bias=True))
            if regularization == 'dropout':
                layers.append(torch.nn.Dropout(0.5))
            elif regularization == 'batchnorm':
                layers.append(torch.nn.BatchNorm1d(neurons))
            elif regularization is None:
                pass

            if activation_type == 'relu':
                layers.append(torch.nn.ReLU(True))
            elif activation_type == 'lrelu':
                layers.append(torch.nn.LeakyReLU(0.3))

        self.forward_part = torch.nn.Sequential(*layers)

    def forward(self, input):
        return input + self.forward_part(input)


class EyeCenteringByRegression:
    def __init__(self, eval_mode=False, make_cuda=False, num_skip_blks=2, intermediate_neurons=512,
                 regularization='batchnorm', num_layers_per_block=2, activation_type='relu'):
        self.mean_input = torch.from_numpy(np.array([ 0.4671627 , -0.09504398, -0.12090819,
                                                      1.2735702 ,  0.00253953, -0.02751609,
                                                      0.10822426, -0.01990774,  0.00626311,
                                                      0.08915882,  0.00973385, -0.00834262]).astype('float32'))
        self.std_input = torch.from_numpy(np.array([0.53506327, 0.52815205, 0.52134556,
                                                    1.1373067 , 0.4865559 , 0.21345851,
                                                    0.11624492, 0.27343082, 0.02041259,
                                                    0.05613742, 0.01074448, 0.03475167]).astype('float32'))
        self.mean_output= torch.from_numpy(np.array([8.0179777e+00,  3.4307071e-03, -1.3698899e-04]).astype('float32'))
        self.std_output = torch.from_numpy(np.array([0.38766932, 0.03351782, 0.01525018]).astype('float32'))
        self.random_model = True

        self.model = torch.nn.Sequential(
            torch.nn.Linear(len(SHAPE + EXP + POSE), intermediate_neurons, bias=True),
            torch.nn.BatchNorm1d(intermediate_neurons),
            torch.nn.ReLU(True),

            *[RegressorNNSkipPart(intermediate_neurons, regularization=regularization,
                                  num_layers_per_block=num_layers_per_block, activation_type=activation_type)
              for skip_blk_id in range(num_skip_blks)],

            torch.nn.Linear(intermediate_neurons, 3, bias=True),
        )

        if make_cuda:
            self.device = 'cuda'
            self.model = self.model.cuda()
        else:
            self.device = 'cpu'

        self.eval_mode = eval_mode
        if eval_mode:
            self.model.eval()

        self.mdl_optim = torch.optim.Adam(self.model.parameters(), lr=1e-4, betas=(0.0, 0.99))
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.mdl_optim, 'min', factor=0.5, patience=5, verbose=True, threshold=0.0001, min_lr=1e-6)

    def load_model(self, checkpoint_path):
        self.random_model = False
        self.model.load_state_dict(torch.load(checkpoint_path))

    def save_model(self, checkpoint_path):
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save(self.model.state_dict(), checkpoint_path)

    def get_camera(self, pose_shape_exp):
        if self.random_model:
            raise ValueError('Using model inference without training or loading it')
        with torch.no_grad():
            self.mean_input = self.mean_input.to(pose_shape_exp.device)
            self.std_input = self.std_input.to(pose_shape_exp.device)
            self.std_output = self.std_output.to(pose_shape_exp.device)
            self.mean_output = self.mean_output.to(pose_shape_exp.device)
            t = (self.model((pose_shape_exp - self.mean_input) / self.std_input) * self.std_output) + self.mean_output
        return t

    def substitute_flame_batch_with_regressed_camera(self, flame_batch):
        t_cam = self.get_camera(flame_batch[:, SHAPE+EXP+POSE])
        flame_batch[:, constants.get_idx_list('TRANS')] = t_cam
        return flame_batch

    def fit_to_data(self, trn_dataloader, epochs=20, verbose=True, training_criterion=torch.nn.MSELoss(),
                    validation_loader=None, save_best_mdl_path=None):
        assert not self.eval_mode
        validation_criterion = torch.nn.MSELoss()
        self.random_model = False
        trn_dataloader_itr = iter(trn_dataloader)
        validation_loss = 0
        best_validation_loss = np.inf
        for epoch_id in range(epochs):
            moving_avg_trn_loss = 0
            self.model.train()
            if verbose:
                pbar = tqdm.tqdm(range(len(trn_dataloader)))
            else:
                pbar = range(len(trn_dataloader))
            for batch_id in pbar:
                try:
                    x_train, y_train = next(trn_dataloader_itr)
                except (OSError, StopIteration):
                    trn_dataloader_itr = iter(trn_dataloader)
                    x_train, y_train = next(trn_dataloader_itr)

                x_train = (x_train - self.mean_input)/self.std_input
                y_train = (y_train - self.mean_output)/self.std_output
                x_train = x_train.to(self.device)
                y_train = y_train.to(self.device)

                # import ipdb;
                # ipdb.set_trace()
                y_hat_train = self.model(x_train)
                train_loss = training_criterion(y_hat_train, y_train)
                train_loss.backward()
                self.mdl_optim.step()
                moving_avg_trn_loss += train_loss.item()
                state_msg = f'[{epoch_id}/{epochs}] Train_loss: {moving_avg_trn_loss/(batch_id+1):.3f} ' \
                            f'Valid_loss: {validation_loss:0.3f}'
                if verbose:
                    pbar.set_description(state_msg)

            # import ipdb; ipdb.set_trace()

            if validation_loader is not None:
                validation_loss = 0
                num_batches = 0
                validation_loader_itr = iter(validation_loader)
                # import ipdb; ipdb.set_trace()
                self.model.eval()
                with torch.no_grad():
                    for x_valid, y_valid in validation_loader_itr:
                        x_valid = (x_valid - self.mean_input) / self.std_input
                        y_valid = (y_valid - self.mean_output) / self.std_output
                        x_valid = x_valid.to(self.device)
                        y_valid = y_valid.to(self.device)

                        num_batches += 1
                        y_hat_valid = self.model(x_valid)
                        valid_loss = validation_criterion(y_hat_valid, y_valid)
                        validation_loss += valid_loss
                validation_loss /= num_batches
                self.lr_scheduler.step(validation_loss)

            if save_best_mdl_path is not None and validation_loader is not None:
                if best_validation_loss > validation_loss:
                    best_validation_loss = validation_loss
                    self.save_model(save_best_mdl_path)
                    print(f'New best model saved to {save_best_mdl_path}')


    def get_eye_center_camera(self, current_shape_exp_pose):
        return self.model(current_shape_exp_pose)


if __name__ == '__main__':
    ''' Regressor training code'''
    from torch.utils.data import Dataset, DataLoader


    class FlmDatLoader(Dataset):

        def __init__(self, keys, param_dict):
            self.param_dict = param_dict
            self.keys = keys
            self.list_bad_images = np.load(cnst.list_deca_failed_iamges)['bad_images']

        def __getitem__(self, index):
            curren_file = str(index).zfill(5) + '.npy'
            while curren_file in self.list_bad_images:
                index = np.random.randint(0, len(self.keys))
                curren_file = str(index).zfill(5) + '.npy'

            shape_exp_pose = np.concatenate((self.param_dict[keys[index]]['shape'][:3],
                                             self.param_dict[keys[index]]['exp'][:3],
                                             self.param_dict[keys[index]]['pose']), axis=-1)

            t_cam = self.param_dict[keys[index]]['cam']

            return shape_exp_pose, t_cam

        def __len__(self):
            return len(self.keys)

    params_dict = np.load(cnst.all_flame_params_file, allow_pickle=True).item()
    keys = []
    for key in params_dict.keys():
        keys.append(key)


    keys = np.array(keys)
    validation_fraction = 0.3
    # import ipdb; ipdb.set_trace()
    train_keys = keys[:int(len(keys) * (1 - validation_fraction))]
    validation_keys = keys[int(len(keys) * (1 - validation_fraction)):]

    train_set = FlmDatLoader(train_keys, params_dict)
    train_loader = DataLoader(train_set, shuffle=True, batch_size=64, num_workers=0, drop_last=True,
                              pin_memory=True)

    valid_set = FlmDatLoader(validation_keys, params_dict)
    validation_loader = DataLoader(valid_set, shuffle=True, batch_size=128, num_workers=0, drop_last=True,
                              pin_memory=True)

    # eye_cntr_reg = EyeCenteringByRegression(num_skip_blks=2, intermediate_neurons=512, regularization='batchnorm',
    #                                         num_layers_per_block=2, activation_type='relu')
    eye_cntr_reg = EyeCenteringByRegression(make_cuda=True, num_skip_blks=2, intermediate_neurons=825,
                                            regularization='batchnorm', num_layers_per_block=1,
                                            activation_type='relu')
    try:
        eye_cntr_reg.fit_to_data(trn_dataloader=train_loader, validation_loader=validation_loader, epochs=200,
                                 save_best_mdl_path='../checkpoint/eye_centering/cntr_flm_param_to_cam.mdl',
                                 training_criterion=torch.nn.MSELoss())
    finally:
        eye_cntr_reg.save_model('../checkpoint/eye_centering/cntr_eye_flm_param_to_cam_last.mdl')
        print('..................Model saved .................')




