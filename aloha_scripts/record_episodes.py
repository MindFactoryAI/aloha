import os
import time
import h5py
import argparse
import h5py_cache
import numpy as np
from tqdm import tqdm

from aloha_scripts.constants import DT, START_ARM_POSE, TASK_CONFIGS, get_start_arm_pose
from aloha_scripts.constants import MASTER_GRIPPER_JOINT_MID, PUPPET_GRIPPER_JOINT_CLOSE, PUPPET_GRIPPER_JOINT_OPEN
from aloha_scripts.constants import PUPPET2MASTER_JOINT_FN
from robot_utils import Recorder, ImageRecorder, get_arm_gripper_positions
from robot_utils import move_arms, torque_on, torque_off, move_grippers, gripper_torque_on, gripper_torque_off
from real_env import make_real_env, get_action
from math import copysign

from interbotix_xs_modules.arm import InterbotixManipulatorXS
import numpy as np

import IPython
e = IPython.embed
from aloha_scripts.robot_utils import reboot_arms, reboot_grippers


def wait_for_start(env, master_bot_left, master_bot_right, close_thresh=0.2):

    master_bot_left.dxl.robot_torque_enable("single", "gripper", True)
    master_bot_right.dxl.robot_torque_enable("single", "gripper", True)

    move_grippers([master_bot_left, master_bot_right], [MASTER_GRIPPER_JOINT_MID, MASTER_GRIPPER_JOINT_MID], 0.2)

    # press gripper to start data collection
    # disable torque for only gripper joint of master robot to allow user movement
    master_bot_left.dxl.robot_torque_enable("single", "gripper", False)
    master_bot_right.dxl.robot_torque_enable("single", "gripper", False)
    print(f'Close the gripper to start')

    pressed = False
    while not pressed:
        d_left = get_arm_gripper_positions(master_bot_left) - MASTER_GRIPPER_JOINT_MID
        d_right = get_arm_gripper_positions(master_bot_right) - MASTER_GRIPPER_JOINT_MID
        print(d_right)
        is_pressed = lambda x: copysign(1., x) < 0
        left_pressed, right_pressed = is_pressed(d_left), is_pressed(d_right)
        left_over_theshold, right_over_threshold = abs(d_left) > close_thresh, abs(d_right) > close_thresh
        if left_pressed and left_over_theshold and right_pressed and right_over_threshold:
            pressed = True
        time.sleep(DT/10)
    master_bot_left.dxl.robot_torque_enable("single", "gripper", True)
    master_bot_right.dxl.robot_torque_enable("single", "gripper", True)

    master_bot_left_pos = PUPPET2MASTER_JOINT_FN(get_arm_gripper_positions(env.puppet_bot_left))
    master_bot_right_pos = PUPPET2MASTER_JOINT_FN(get_arm_gripper_positions(env.puppet_bot_right))
    move_grippers([master_bot_left, master_bot_right], [master_bot_left_pos, master_bot_right_pos], 0.2)
    print(f'Started!')


def opening_ceremony(master_bot_left, master_bot_right, puppet_bot_left, puppet_bot_right, reboot, current_limit, start_left_arm_pose, start_right_arm_pose):
    if reboot:
        reboot_arms(master_bot_left, master_bot_right, puppet_bot_left, puppet_bot_right, current_limit)
    else:  # rebooting the grippers each episode as they often operate past their current limit
        reboot_grippers(puppet_bot_left, puppet_bot_right, current_limit)

    torque_on(puppet_bot_left)
    torque_on(master_bot_left)
    torque_on(puppet_bot_right)
    torque_on(master_bot_right)

    # move arms to starting position
    move_arms([master_bot_left, puppet_bot_left, master_bot_right, puppet_bot_right], [start_left_arm_pose] * 2 + [start_right_arm_pose] * 2, move_time=1.5)
    # move grippers to starting position
    move_grippers([master_bot_left, puppet_bot_left, master_bot_right, puppet_bot_right], [MASTER_GRIPPER_JOINT_MID, PUPPET_GRIPPER_JOINT_CLOSE] * 2, move_time=0.5)


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


def teleoperate(states, actions, actual_dt_history, max_timesteps, env, master_bot_left, master_bot_right):
    torque_off(master_bot_left)
    torque_off(master_bot_right)
    for t in tqdm(range(max_timesteps)):
        t0 = time.time() #
        action = get_action(master_bot_left, master_bot_right)
        actions.append(action)
        t1 = time.time() #
        state = env.step(action)
        t2 = time.time() #
        states.append(state)
        actual_dt_history.append([t0, t1, t2])
    torque_on(master_bot_left)
    torque_on(master_bot_right)
    return states, actions, actual_dt_history


def save_episode(dataset_path, camera_names, max_timesteps, timesteps, actions):
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
    """

    data_dict = {
        '/observations/qpos': [],
        '/observations/qvel': [],
        '/observations/effort': [],
        '/action': [],
    }
    for cam_name in camera_names:
        data_dict[f'/observations/images/{cam_name}'] = []

    # len(action): max_timesteps, len(time_steps): max_timesteps + 1
    assert len(timesteps) == max_timesteps + 1, f"expected {max_timesteps + 1} timesteps got {len(timesteps)}"
    assert len(actions) == max_timesteps, f"expected {max_timesteps} actions got {len(actions)}"
    while actions:
        action = actions.pop(0)
        state = timesteps.pop(0)
        data_dict['/observations/qpos'].append(state.observation['qpos'])
        data_dict['/observations/qvel'].append(state.observation['qvel'])
        data_dict['/observations/effort'].append(state.observation['effort'])
        data_dict['/action'].append(action)
        for cam_name in camera_names:
            # we are going to add the compressed images to the dataset
            data_dict[f'/observations/images/{cam_name}'].append(
                state.observation['images_compressed'][cam_name])

    # HDF5
    t0 = time.time()
    # import h5py_cache
    # with h5py_cache.File(dataset_path + '.hdf5', 'w', chunk_cache_mem_size=1024**2*2) as root:
    with h5py.File(dataset_path + '.hdf5', 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
        root.attrs['sim'] = False
        root.attrs['episode_len'] = max_timesteps
        obs = root.create_group('observations')
        images = obs.create_group('images')

        for cam_name in camera_names:
            cam_g = images.create_group(cam_name)
            image_list = data_dict[f'/observations/images/{cam_name}']
            filenames = [f'{i}.jpg' for i, _ in enumerate(data_dict[f'/observations/images/{cam_name}'])]
            for fname, image in zip(filenames, image_list):
                cam_g.create_dataset(fname, data=np.void(image))

        for name in ['/observations/qpos', '/observations/qvel', '/observations/effort', '/action']:
            _ = obs.create_dataset(name, (max_timesteps, 14))
            root[name][...] = data_dict[name]

    print(f'Saving: {time.time() - t0:.1f} secs')

    return True


def capture_one_episode(initial_state, dt, max_timesteps, camera_names, dataset_dir, dataset_name, overwrite,
                        master_bot_left, master_bot_right, env):

    dataset_path = validate_dataset(dataset_dir, dataset_name, overwrite)
    states, actions, actual_dt_history = [initial_state], [], []
    timesteps, actions, actual_dt_history = teleoperate(states, actions, actual_dt_history, max_timesteps, env, master_bot_left, master_bot_right)

    # Torque on both master bots
    torque_on(master_bot_left)
    torque_on(master_bot_right)

    # Open puppet grippers
    env.puppet_bot_left.dxl.robot_set_operating_modes("single", "gripper", "position")
    env.puppet_bot_right.dxl.robot_set_operating_modes("single", "gripper", "position")
    move_grippers([env.puppet_bot_left, env.puppet_bot_right], [PUPPET_GRIPPER_JOINT_OPEN] * 2, move_time=0.5)
    env.puppet_bot_left.dxl.robot_set_operating_modes("single", "gripper", "current_based_position")
    env.puppet_bot_right.dxl.robot_set_operating_modes("single", "gripper", "current_based_position")

    freq_mean = print_dt_diagnosis(actual_dt_history)
    if freq_mean < 42:
        return False

    did_save = save_episode(dataset_path, camera_names, max_timesteps, timesteps, actions)
    return did_save



def main(args):
    task_config = TASK_CONFIGS[args['task_name']]
    dataset_dir = task_config['dataset_dir']
    max_timesteps = task_config['episode_len']
    camera_names = task_config['camera_names']
    current_limit = args['current_limit']

    # source of data
    master_bot_left = InterbotixManipulatorXS(robot_model="wx250s", group_name="arm", gripper_name="gripper",
                                              robot_name=f'master_left', init_node=True)
    master_bot_right = InterbotixManipulatorXS(robot_model="wx250s", group_name="arm", gripper_name="gripper",
                                               robot_name=f'master_right', init_node=False)
    env = make_real_env(init_node=False, setup_robots=False, task=args['task_name'])

    reboot = True  # always reboot on the first episode

    while True:

        if args['episode_idx'] is not None:
            episode_idx = args['episode_idx']
        else:
            episode_idx = get_auto_index(dataset_dir)
        overwrite = True
        dataset_name = f'episode_{episode_idx}'
        print(dataset_name + '\n')
        start_left_arm_pose, start_right_arm_pose = get_start_arm_pose(args['task_name'])
        # move all 4 robots to a starting pose where it is easy to start teleoperation
        opening_ceremony(master_bot_left, master_bot_right, env.puppet_bot_left, env.puppet_bot_right, reboot,
                         current_limit, start_left_arm_pose, start_right_arm_pose)
        # then wait till both gripper closed
        wait_for_start(env, master_bot_left, master_bot_right)
        ts = env.reset(fake=True)
        is_healthy = capture_one_episode(ts, DT, max_timesteps, camera_names, dataset_dir, dataset_name, overwrite,
                                         master_bot_left, master_bot_right, env)
        if is_healthy and args['episode_idx'] is not None:
            break
        reboot = args['reboot_every_episode']


def get_auto_index(dataset_dir, dataset_name_prefix = '', data_suffix = 'hdf5'):
    max_idx = 1000
    if not os.path.isdir(dataset_dir):
        os.makedirs(dataset_dir)
    for i in range(max_idx+1):
        if not os.path.isfile(os.path.join(dataset_dir, f'{dataset_name_prefix}episode_{i}.{data_suffix}')):
            return i
    raise Exception(f"Error getting auto index, or more than {max_idx} episodes")


def print_dt_diagnosis(actual_dt_history):
    actual_dt_history = np.array(actual_dt_history)
    get_action_time = actual_dt_history[:, 1] - actual_dt_history[:, 0]
    step_env_time = actual_dt_history[:, 2] - actual_dt_history[:, 1]
    total_time = actual_dt_history[:, 2] - actual_dt_history[:, 0]

    dt_mean = np.mean(total_time)
    dt_std = np.std(total_time)
    freq_mean = 1 / dt_mean
    print(f'Avg freq: {freq_mean:.2f} Get action: {np.mean(get_action_time):.3f} Step env: {np.mean(step_env_time):.3f}')
    return freq_mean


def debug():
    print(f'====== Debug mode ======')
    recorder = Recorder('right', is_debug=True)
    image_recorder = ImageRecorder(init_node=False, is_debug=True)
    while True:
        time.sleep(1)
        recorder.print_diagnostics()
        image_recorder.print_diagnostics()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', action='store', type=str, help='Task name.', required=True)
    parser.add_argument('--episode_idx', action='store', type=int, help='Episode index.', default=None, required=False)
    parser.add_argument('--reboot_every_episode', action='store_true', help='Episode index.', default=False, required=False)
    parser.add_argument('--current_limit', help='Gripper current limit.', type=int, default=300, required=False)
    main(vars(parser.parse_args()))
    # debug()


