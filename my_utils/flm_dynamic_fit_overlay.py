import torch
import numpy as np


def camera_ringnet(cam):
    camera_params = {'c': cam[1:3],
                     'k': np.zeros(5),
                     'f': cam[0] * np.ones(2)}
    camera_params['t'] = np.zeros(3)
    camera_params['r'] = np.zeros(3)
    return camera_params


def camera_dynamic(h_w, translation):
    h, w = h_w
    fscale = h / 256
    camera_params = {'c': np.array([w / 2, h / 2]),
                     'k': np.array([-0.19816071, 0.92822711, 0, 0, 0]),
                     'f': np.array([fscale*4754.97941935, fscale*4754.97941935])}
    camera_params['t'] = translation
    camera_params['r'] = np.array([np.pi, 0., 0.])
    return camera_params


def camera_ringnetpp(h_w, trans, focal):
    h, w = h_w
    camera_params = {'c': np.array([w / 2, h / 2]),
                     'k': np.zeros(5),
                     'f': focal * np.ones(2)}
    camera_params['t'] = trans
    camera_params['r'] = np.array([0., np.pi, 0.])
    return camera_params


if __name__ == '__main__':
    # config = get_config()
    # ##
    # config.model_name = 'optimize_flame'
    # config.resume_training = True
    # config.batch_size = 1
    # config.dataset_path = {
    #     'vgg2': 'dataset_loaders/vggface2_train_list_max_normal_100_ring_3_3_serial.npy'
    #     }
    # config.ring_elements = 1
    #
    # # generate
    # # generate_rendering(config)
    # camera_test(config)

    cam_t = np.array([0., 0., 0.]) + np.array([0., 0., 2.5])
    camera_params = camera_dynamic((256, 256), cam_t)

    points_3d = torch.from_numpy(np.random.uniform(-1, 1, (32, 4, 3)).astype('float32')).cuda()
    points_2d = batch_perspective_proj(points_3d, camera_params)
    print(points_2d.shape)