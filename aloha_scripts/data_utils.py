import os
import time

import cv2
import h5py
import numpy as np
from pathlib import Path


def save_episode(dataset_path, policy_guid, camera_names, max_timesteps, timesteps, actions, terminal_state=None, result=None, policy_info=None, policy_index=None):
    """
    For each timestep:
    observations
    - images
        - cam_high          (480, 640, 3) 'uint8'
        - cam_low           (480, 640, 3) 'uint8'
        - cam_left_wrist    (480, 640, 3) 'uint8'
        - cam_right_wrist   (480, 640, 3) 'uint8'
    - qpos                  (14,)         'float64'
    - qvel                  (14,)         'float64'

    action                  (14,)         'float64'

    policy_guid: the guid of the policy that created this episode, "human" if human
    """

    if len(timesteps) == max_timesteps + 1 and terminal_state is None:
        terminal_state = timesteps[-1]
    elif terminal_state is None:
        raise Exception('terminal state is missing, either explicity set it using terminal_state or add an extra state to the state buffer')
    assert len(actions) == max_timesteps, f"expected {max_timesteps} actions got {len(actions)}"

    data_dict = {
        '/observations/qpos': [],
        '/observations/qvel': [],
        '/observations/effort': [],
        '/action': [],
    }
    for cam_name in camera_names:
        data_dict[f'/observations/images/{cam_name}'] = []

    while actions:
        action = actions.pop(0)
        state = timesteps.pop(0)
        data_dict['/observations/qpos'].append(state.observation['qpos'])
        data_dict['/observations/qvel'].append(state.observation['qvel'])
        data_dict['/observations/effort'].append(state.observation['effort'])
        data_dict['/action'].append(action)
        if policy_index is not None:
            data_dict['/policy'].append(policy_index.pop(0))
        for cam_name in camera_names:
            # we are going to add the compressed images to the dataset
            data_dict[f'/observations/images/{cam_name}'].append(
                state.observation['images_compressed'][cam_name])

    # add terminal state
    data_dict['/observations/qpos'].append(terminal_state.observation['qpos'])
    data_dict['/observations/qvel'].append(terminal_state.observation['qvel'])
    data_dict['/observations/effort'].append(terminal_state.observation['effort'])
    for cam_name in camera_names:
        data_dict[f'/observations/images/{cam_name}'].append(
            terminal_state.observation['images_compressed'][cam_name])

    # HDF5
    t0 = time.time()
    # import h5py_cache
    # with h5py_cache.File(dataset_path + '.hdf5', 'w', chunk_cache_mem_size=1024**2*2) as root:
    with h5py.File(dataset_path + '.hdf5', 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
        root.attrs['policy_guid'] = policy_guid
        root.attrs['sim'] = False
        root.attrs['episode_len'] = max_timesteps
        if result is not None:
            root.attrs['result'] = result
        obs = root.create_group('observations')
        images = obs.create_group('images')

        for cam_name in camera_names:
            cam_g = images.create_group(cam_name)
            image_list = data_dict[f'/observations/images/{cam_name}']
            filenames = [f'{i}.jpg' for i, _ in enumerate(data_dict[f'/observations/images/{cam_name}'])]
            for fname, image in zip(filenames, image_list):
                cam_g.create_dataset(fname, data=np.void(image))

        for name in ['/observations/qpos', '/observations/qvel', '/observations/effort']:
            _ = obs.create_dataset(name, (max_timesteps+1, 14))
            root[name][...] = data_dict[name]

        for name in ['/action']:
            _ = obs.create_dataset(name, (max_timesteps, 14))
            root[name][...] = data_dict[name]

        if policy_info is not None:
            root.create_dataset('policy_info', data=np.array(policy_info, dtype='S'))

        if policy_index is not None:
            root.create_dataset('policy', data=np.array(policy_index))

    print(f'Saving: {time.time() - t0:.1f} secs')

    return True


def validate_dataset(dataset_dir, dataset_name, overwrite=False):
    print(f'Dataset name: {dataset_name}')
    # saving dataset
    if not os.path.isdir(dataset_dir):
        os.makedirs(dataset_dir)
    dataset_path = os.path.join(dataset_dir, dataset_name)
    if os.path.isfile(dataset_path) and not overwrite:
        print(f'Dataset already exist at \n{dataset_path}\nHint: set overwrite to True.')
        exit()
    return dataset_path


def get_auto_index(dataset_dir, dataset_name_prefix = '', data_suffix = 'hdf5'):
    max_idx = 1000
    if not os.path.isdir(dataset_dir):
        os.makedirs(dataset_dir)
    for i in range(max_idx+1):
        if not os.path.isfile(os.path.join(dataset_dir, f'{dataset_name_prefix}episode_{i}.{data_suffix}')):
            return i
    raise Exception(f"Error getting auto index, or more than {max_idx} episodes")


def load_hdf5(dataset_dir, dataset_name):
    dataset_path = os.path.join(dataset_dir, dataset_name + '.hdf5')
    if not os.path.isfile(dataset_path):
        print(f'Dataset does not exist at \n{dataset_path}\n')
        exit()

    with h5py.File(dataset_path, 'r') as root:
        timesteps = root.attrs['episode_len']
        is_sim = root.attrs['sim']
        qpos = root['/observations/qpos'][()]
        qvel = root['/observations/qvel'][()]
        effort = root['/observations/effort'][()]
        action = root['/action'][()]
        image_dict = dict()
        for cam_name in root[f'/observations/images/'].keys():
            images = []
            for i in range(timesteps):
                image_bytes = root[f'/observations/images/{cam_name}/{i}.jpg'][()]
                image_rgb = decompress_image(image_bytes)
                images.append(image_rgb)
            image_dict[cam_name] = images

    return qpos, qvel, effort, action, image_dict


def decompress_image(image_bytes, format='BGR'):
    """
    The robot outputs BGR format natively, so we will use that, convert to RGB for display purposes
    """
    image = np.frombuffer(image_bytes, np.uint8)
    image_bgr = cv2.imdecode(image, cv2.IMREAD_COLOR)
    if format == 'BGR':
        return image_bgr
    elif format == 'RGB':
        return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    else:
        raise Exception(f"{format} not supported, convert from BGR to your required format")


class Dataset:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path

    def get_newest_episode(self):
        files = Path(self.dataset_path).glob('*.hdf5')  # * means all if need specific format then *.csv
        latest_file = max(list(files), key=lambda x: x.stat().st_ctime)
        return Episode(latest_file)


class Episode:
    def __init__(self, episode_path):
        self.episode_path = episode_path

    def get_initial_frame(self, format='BGR', return_type='image'):
        return self.get_frame(0, format, return_type)

    def get_terminal_frame(self, format='BGR', return_type='image'):
        return self.get_frame(len(self)-1, format, return_type) # todo: this doens't return the exact terminal frame

    def get_frame(self, index, format='BGR', return_type='image'):
        cam_frames = {}
        with h5py.File(self.episode_path, 'r') as root:
            for cam_name in root[f'/observations/images/'].keys():
                cam_frames[cam_name] = decompress_image(root[f'/observations/images/{cam_name}/{index}.jpg'][()], format=format)
            if return_type == 'dict':
                return cam_frames
            else:
                return np.concatenate(list(cam_frames.values()), axis=1)

    def get_cam_names(self):
        with h5py.File(self.episode_path, 'r') as root:
            return [cam_name for cam_name in root[f'/observations/images/'].keys()]

    def split_frame(self, frame):
        cams = {}
        for i, cam_name in enumerate(self.get_cam_names()):
            cams[cam_name] = frame[:, i * 640:(i+1) * 640]
        return cams

    def __len__(self):
        with h5py.File(self.episode_path, 'r') as root:
            return root.attrs['episode_len']