import time
import argparse
from tqdm import tqdm

from aloha_scripts.constants import DT, TASK_CONFIGS, get_start_arm_pose
from aloha_scripts.constants import MASTER_GRIPPER_JOINT_MID, PUPPET_GRIPPER_JOINT_CLOSE, PUPPET_GRIPPER_JOINT_OPEN
from data_utils import save_episode, validate_dataset, get_auto_index
from robot_utils import Recorder, ImageRecorder, wait_for_input
from robot_utils import move_arms, torque_on, torque_off, move_grippers
from real_env import make_real_env, get_action

from interbotix_xs_modules.arm import InterbotixManipulatorXS
import numpy as np

import IPython
e = IPython.embed
from aloha_scripts.robot_utils import reboot_arms, reboot_grippers


def opening_ceremony(master_bot_left, master_bot_right, puppet_bot_left, puppet_bot_right, reboot, current_limit, start_left_arm_pose, start_right_arm_pose):
    if reboot:
        reboot_arms(master_bot_left, master_bot_right, puppet_bot_left, puppet_bot_right, current_limit)
    else:  # rebooting the grippers each episode as they often operate past their current limit
        reboot_grippers(master_bot_left, master_bot_right, puppet_bot_left, puppet_bot_right, current_limit)

    torque_on(puppet_bot_left)
    torque_on(master_bot_left)
    torque_on(puppet_bot_right)
    torque_on(master_bot_right)

    # move arms to starting position
    move_arms([master_bot_left, puppet_bot_left, master_bot_right, puppet_bot_right], [start_left_arm_pose] * 2 + [start_right_arm_pose] * 2, move_time=1.5)
    # move grippers to starting position
    move_grippers([master_bot_left, puppet_bot_left, master_bot_right, puppet_bot_right], [MASTER_GRIPPER_JOINT_MID, PUPPET_GRIPPER_JOINT_CLOSE] * 2, move_time=0.5)


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
        wait_for_input(env, master_bot_left, master_bot_right)
        ts = env.reset(fake=True)
        is_healthy = capture_one_episode(ts, DT, max_timesteps, camera_names, dataset_dir, dataset_name, overwrite,
                                         master_bot_left, master_bot_right, env)
        if is_healthy and args['episode_idx'] is not None:
            break
        reboot = args['reboot_every_episode']


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


