import os
import h5py
from robot_utils import move_grippers
import argparse
from real_env import make_real_env
from constants import JOINT_NAMES, PUPPET_GRIPPER_JOINT_OPEN

import IPython
e = IPython.embed

STATE_NAMES = JOINT_NAMES + ["gripper", 'left_finger', 'right_finger']


def replay(dataset_path):
    with h5py.File(dataset_path, 'r') as root:
        actions = root['/action'][()]

    env = make_real_env(init_node=True)
    env.reset()
    for action in actions:
        env.step(action)

    move_grippers([env.puppet_bot_left, env.puppet_bot_right], [PUPPET_GRIPPER_JOINT_OPEN] * 2, move_time=0.5)  # open


def main(args):

    if not os.path.isfile(args.filename):
        print(f'Dataset does not exist at \n{args.filename}\n')
        exit()

    replay(args.filename)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('filename', action='store', type=str, help='file to replay')
    args = parser.parse_args()
    main(args)


