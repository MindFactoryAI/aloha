### Task parameters


CAM_NAMES = ['cam_high', 'cam_low', 'cam_left_wrist', 'cam_right_wrist']
DATA_DIR = '/mnt/magneto/data/aloha_recordings'
FAST_DATA_DIR = '/home/duane/data'

TASK_CONFIGS = {
    'move_tape':{
        'dataset_dir': DATA_DIR + '/move_tape',
        'num_episodes': 100,
        'episode_len': 1000,
        'camera_names': ['cam_high', 'cam_low', 'cam_left_wrist', 'cam_right_wrist']
    },
    'move_durex': {
        'dataset_dir': DATA_DIR + '/move_durex',
        'num_episodes': 100,
        'episode_len': 1000,
        'camera_names': ['cam_high', 'cam_low', 'cam_left_wrist', 'cam_right_wrist']
    },
    'move_durex_reverse': {
        'dataset_dir': DATA_DIR + '/move_durex_reverse',
        'num_episodes': 50,
        'episode_len': 500,
        'camera_names': ['cam_high', 'cam_low', 'cam_left_wrist', 'cam_right_wrist']
    },
    'grasp_condi': {
        'dataset_dir': DATA_DIR + '/grasp_condi',
        'num_episodes': 50,
        'episode_len': 500,
        'camera_names': ['cam_high', 'cam_low', 'cam_left_wrist', 'cam_right_wrist']
    },
    'grasp_condi_2': {
        'dataset_dir': FAST_DATA_DIR + '/grasp_condi_2',
        'num_episodes': 100,
        'episode_len': 600,
        'camera_names': ['cam_high', 'cam_low', 'cam_left_wrist', 'cam_right_wrist']
    },
    'touch_condi': {
        'dataset_dir': DATA_DIR + '/touch_condi',
        'num_episodes': 50,
        'episode_len': 200,
        'camera_names': ['cam_high', 'cam_low', 'cam_left_wrist', 'cam_right_wrist']
    },
    'touch_condi_compressed': {
        'dataset_dir': FAST_DATA_DIR + '/touch_condi',
        'num_episodes': 50,
        'episode_len': 200,
        'camera_names': ['cam_high', 'cam_low', 'cam_left_wrist', 'cam_right_wrist']
    },
    'pop_condi': {
        'dataset_dir': DATA_DIR + '/pop_condi',
        'num_episodes': 50,
        'episode_len': 700,
        'camera_names': ['cam_high', 'cam_low', 'cam_left_wrist', 'cam_right_wrist']
    },
    'pop_condi_compressed_50': {
        'dataset_dir': FAST_DATA_DIR + '/pop_condi',
        'num_episodes': 50,
        'episode_len': 700,
        'camera_names': ['cam_high', 'cam_low', 'cam_left_wrist', 'cam_right_wrist']
    },
    'pop_condi_100': {
        'dataset_dir': DATA_DIR + '/pop_condi',
        'num_episodes': 100,
        'episode_len': 700,
        'camera_names': ['cam_high', 'cam_low', 'cam_left_wrist', 'cam_right_wrist']
    },
    'pop_condi_compressed_100': {
        'dataset_dir': FAST_DATA_DIR + '/pop_condi',
        'num_episodes': 100,
        'episode_len': 700,
        'camera_names': ['cam_high', 'cam_low', 'cam_left_wrist', 'cam_right_wrist']
    },
    'pop_condi_compressed_175': {
        'dataset_dir': FAST_DATA_DIR + '/pop_condi',
        'num_episodes': 175,
        'episode_len': 700,
        'camera_names': ['cam_high', 'cam_low', 'cam_left_wrist', 'cam_right_wrist']
    },
    'pick_ziplock': {
        'dataset_dir': DATA_DIR + '/pick_ziplock',
        'num_episodes': 50,
        'episode_len': 300,
        'camera_names': ['cam_high', 'cam_low', 'cam_left_wrist', 'cam_right_wrist']
    },
    'pick_ziplock_compressed': {
        'dataset_dir': FAST_DATA_DIR + '/pick_ziplock',
        'num_episodes': 50,
        'episode_len': 300,
        'camera_names': ['cam_high', 'cam_low', 'cam_left_wrist', 'cam_right_wrist']
    },
    'tip_cup': {
        'dataset_dir': DATA_DIR + '/tip_cup',
        'num_episodes': 50,
        'episode_len': 300,
        'camera_names': ['cam_high', 'cam_low', 'cam_left_wrist', 'cam_right_wrist']
    },
    'tip_cup_compressed': {
        'dataset_dir': FAST_DATA_DIR + '/tip_cup',
        'num_episodes': 50,
        'episode_len': 300,
        'camera_names': ['cam_high', 'cam_low', 'cam_left_wrist', 'cam_right_wrist']
    },
    'pop_cup': {
        'dataset_dir': DATA_DIR + '/pop_cup',
        'num_episodes': 50,
        'episode_len': 700,
        'camera_names': ['cam_high', 'cam_low', 'cam_left_wrist', 'cam_right_wrist']
    },
    'pop_cup_posed': {
        'dataset_dir': DATA_DIR + '/pop_cup_posed',
        'num_episodes': 52,
        'episode_len': 700,
        'camera_names': ['cam_high', 'cam_low', 'cam_left_wrist', 'cam_right_wrist'],
        'start_left_arm_pose': [0.2208932340145111, -0.37889325618743896, 1.2686021327972412, 0.44025251269340515, -0.6135923266410828, -0.2178252786397934],
        'start_right_arm_pose': [-0.14726215600967407, -0.5599030256271362, 1.3023496866226196, -0.058291271328926086, -0.3436117172241211, 0.02147573232650757]
    },
    'pop_cup_slick': {
        'dataset_dir': DATA_DIR + '/pop_cup_slick',
        'num_episodes': 52,
        'episode_len': 700,
        'camera_names': ['cam_high', 'cam_low', 'cam_left_wrist', 'cam_right_wrist'],
        'start_left_arm_pose': [0.2208932340145111, -0.37889325618743896, 1.2686021327972412, 0.44025251269340515,
                                -0.6135923266410828, -0.2178252786397934],
        'start_right_arm_pose': [-0.14726215600967407, -0.5599030256271362, 1.3023496866226196, -0.058291271328926086,
                                 -0.3436117172241211, 0.02147573232650757]
    },
    'pop_cup_slick_mjpeg': {
        'dataset_dir': FAST_DATA_DIR + '/pop_cup_slick_mjpeg',
        'num_episodes': 100,
        'episode_len': 700,
        'camera_names': ['cam_high', 'cam_low', 'cam_left_wrist', 'cam_right_wrist'],
        'start_left_arm_pose': [0.2208932340145111, -0.37889325618743896, 1.2686021327972412, 0.44025251269340515,
                                -0.6135923266410828, -0.2178252786397934],
        'start_right_arm_pose': [-0.14726215600967407, -0.5599030256271362, 1.3023496866226196, -0.058291271328926086,
                                 -0.3436117172241211, 0.02147573232650757]
    },
    'pop_cup_slick_100': {
        'dataset_dir': DATA_DIR + '/pop_cup_slick',
        'num_episodes': 100,
        'episode_len': 700,
        'camera_names': ['cam_high', 'cam_low', 'cam_left_wrist', 'cam_right_wrist'],
        'start_left_arm_pose': [0.2208932340145111, -0.37889325618743896, 1.2686021327972412, 0.44025251269340515,
                                -0.6135923266410828, -0.2178252786397934],
        'start_right_arm_pose': [-0.14726215600967407, -0.5599030256271362, 1.3023496866226196, -0.058291271328926086,
                                 -0.3436117172241211, 0.02147573232650757]
    },
    'slot_battery': {
        'dataset_dir': FAST_DATA_DIR + '/slot_battery',
        'num_episodes': 50,
        'episode_len': 800,
        'camera_names': ['cam_high', 'cam_low', 'cam_left_wrist', 'cam_right_wrist'],
        'start_left_arm_pose': [0.2208932340145111, -0.37889325618743896, 1.2686021327972412, 0.44025251269340515,
                                -0.6135923266410828, -0.2178252786397934],
        'start_right_arm_pose': [-0.14726215600967407, -0.5599030256271362, 1.3023496866226196, -0.058291271328926086,
                                 -0.3436117172241211, 0.02147573232650757]
    },
    'push_battery_in_slot': {
        'dataset_dir': FAST_DATA_DIR + '/push_battery_in_slot',
        'num_episodes': 54,
        'episode_len': 350,
        'camera_names': ['cam_high', 'cam_low', 'cam_left_wrist', 'cam_right_wrist'],
        'start_left_arm_pose': [0.2208932340145111, -0.37889325618743896, 1.2686021327972412, 0.44025251269340515,
                                -0.6135923266410828, -0.2178252786397934],
        'start_right_arm_pose': [-0.14726215600967407, -0.5599030256271362, 1.3023496866226196, -0.058291271328926086,
                                 -0.3436117172241211, 0.02147573232650757]
    },
    'drop_battery_in_slot': {
        'dataset_dir': FAST_DATA_DIR + '/drop_battery_in_slot',
        'num_episodes': 83,
        'episode_len': 350,
        'camera_names': ['cam_high', 'cam_low', 'cam_left_wrist', 'cam_right_wrist'],
        'start_left_arm_pose': [0.2208932340145111, -0.37889325618743896, 1.2686021327972412, 0.44025251269340515,
                                -0.6135923266410828, -0.2178252786397934],
        'start_right_arm_pose': [-0.14726215600967407, -0.5599030256271362, 1.3023496866226196, -0.058291271328926086,
                                 -0.3436117172241211, 0.02147573232650757]
    },
    'drop_battery_in_slot_right': {
        'dataset_dir': FAST_DATA_DIR + '/drop_battery_in_slot_right',
        'num_episodes': 200,
        'episode_len': 450,
        'camera_names': ['cam_high', 'cam_low', 'cam_left_wrist', 'cam_right_wrist'],
        'start_left_arm_pose': [0.2208932340145111, -0.37889325618743896, 1.2686021327972412, 0.44025251269340515,
                                -0.6135923266410828, -0.2178252786397934],
        'start_right_arm_pose': [-0.14726215600967407, -0.5599030256271362, 1.3023496866226196, -0.058291271328926086,
                                 -0.3436117172241211, 0.02147573232650757]
    },
    'grasp_battery': {
        'dataset_dir': FAST_DATA_DIR + '/grasp_battery',
        'num_episodes': 203,
        'episode_len': 200,
        'camera_names': ['cam_high', 'cam_low', 'cam_left_wrist', 'cam_right_wrist'],
        'start_left_arm_pose': [0.2208932340145111, -0.37889325618743896, 1.2686021327972412, 0.44025251269340515,
                                -0.6135923266410828, -0.2178252786397934],
        'start_right_arm_pose': [-0.14726215600967407, -0.5599030256271362, 1.3023496866226196, -0.058291271328926086,
                                 -0.3436117172241211, 0.02147573232650757]
    },
    'drop_battery_in_slot_only': {
        'dataset_dir': FAST_DATA_DIR + '/drop_battery_in_slot_only',
        'num_episodes': 50,
        'episode_len': 350,
        'camera_names': ['cam_high', 'cam_low', 'cam_left_wrist', 'cam_right_wrist'],
        'start_left_arm_pose': [0.2208932340145111, -0.37889325618743896, 1.2686021327972412, 0.44025251269340515,
                                -0.6135923266410828, -0.2178252786397934],
        'start_right_arm_pose': [-0.14726215600967407, -0.5599030256271362, 1.3023496866226196, -0.058291271328926086,
                                 -0.3436117172241211, 0.02147573232650757],

    },
    'slot_battery_long': {
        'dataset_dir': FAST_DATA_DIR + '/slot_battery_long',
        'num_episodes': 50,
        'episode_len': 350,
        'camera_names': CAM_NAMES,
        'start_left_arm_pose': [0.2208932340145111, -0.37889325618743896, 1.2686021327972412, 0.44025251269340515,
                                -0.6135923266410828, -0.2178252786397934],
        'start_right_arm_pose': [-0.14726215600967407, -0.5599030256271362, 1.3023496866226196, -0.058291271328926086,
                                 -0.3436117172241211, 0.02147573232650757],
    }
}


def get_start_arm_pose(task=None):
    if task is not None:
        if 'start_left_arm_pose' in TASK_CONFIGS[task]:
            return TASK_CONFIGS[task]['start_left_arm_pose'], TASK_CONFIGS[task]['start_right_arm_pose']
    return START_ARM_POSE[:6], START_ARM_POSE[:6]


### ALOHA fixed constants
DT = 0.02
JOINT_NAMES = ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"]
START_ARM_POSE = [0, -0.96, 1.16, 0, -0.3, 0, 0.02239, -0.02239,  0, -0.96, 1.16, 0, -0.3, 0, 0.02239, -0.02239]
# Left finger position limits (qpos[7]), right_finger = -1 * left_finger
MASTER_GRIPPER_POSITION_OPEN = 0.02417
MASTER_GRIPPER_POSITION_CLOSE = 0.01244
PUPPET_GRIPPER_POSITION_OPEN = 0.05800
PUPPET_GRIPPER_POSITION_CLOSE = 0.01844

# Gripper joint limits (qpos[6])
MASTER_GRIPPER_JOINT_OPEN = 0.3083
MASTER_GRIPPER_JOINT_CLOSE = -0.6842
PUPPET_GRIPPER_JOINT_OPEN = 1.4910
# PUPPET_GRIPPER_JOINT_CLOSE = -0.6013 # caused motor to overload
# PUPPET_GRIPPER_JOINT_CLOSE = -0.2513 # safe for a weak close
PUPPET_GRIPPER_JOINT_CLOSE = -0.313 # try for stronker close


############################ Helper functions ############################

MASTER_GRIPPER_POSITION_NORMALIZE_FN = lambda x: (x - MASTER_GRIPPER_POSITION_CLOSE) / (MASTER_GRIPPER_POSITION_OPEN - MASTER_GRIPPER_POSITION_CLOSE)
PUPPET_GRIPPER_POSITION_NORMALIZE_FN = lambda x: (x - PUPPET_GRIPPER_POSITION_CLOSE) / (PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE)
MASTER_GRIPPER_POSITION_UNNORMALIZE_FN = lambda x: x * (MASTER_GRIPPER_POSITION_OPEN - MASTER_GRIPPER_POSITION_CLOSE) + MASTER_GRIPPER_POSITION_CLOSE
PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN = lambda x: x * (PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE) + PUPPET_GRIPPER_POSITION_CLOSE
MASTER2PUPPET_POSITION_FN = lambda x: PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(MASTER_GRIPPER_POSITION_NORMALIZE_FN(x))

MASTER_GRIPPER_JOINT_NORMALIZE_FN = lambda x: (x - MASTER_GRIPPER_JOINT_CLOSE) / (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE)
PUPPET_GRIPPER_JOINT_NORMALIZE_FN = lambda x: (x - PUPPET_GRIPPER_JOINT_CLOSE) / (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE)
MASTER_GRIPPER_JOINT_UNNORMALIZE_FN = lambda x: x * (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE) + MASTER_GRIPPER_JOINT_CLOSE
PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN = lambda x: x * (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE) + PUPPET_GRIPPER_JOINT_CLOSE
MASTER2PUPPET_JOINT_FN = lambda x: PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN(MASTER_GRIPPER_JOINT_NORMALIZE_FN(x))
PUPPET2MASTER_JOINT_FN = lambda x: MASTER_GRIPPER_JOINT_UNNORMALIZE_FN(PUPPET_GRIPPER_JOINT_NORMALIZE_FN(x))

MASTER_GRIPPER_VELOCITY_NORMALIZE_FN = lambda x: x / (MASTER_GRIPPER_POSITION_OPEN - MASTER_GRIPPER_POSITION_CLOSE)
PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN = lambda x: x / (PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE)

MASTER_POS2JOINT = lambda x: MASTER_GRIPPER_POSITION_NORMALIZE_FN(x) * (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE) + MASTER_GRIPPER_JOINT_CLOSE
MASTER_JOINT2POS = lambda x: MASTER_GRIPPER_POSITION_UNNORMALIZE_FN((x - MASTER_GRIPPER_JOINT_CLOSE) / (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE))
PUPPET_POS2JOINT = lambda x: PUPPET_GRIPPER_POSITION_NORMALIZE_FN(x) * (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE) + PUPPET_GRIPPER_JOINT_CLOSE
PUPPET_JOINT2POS = lambda x: PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN((x - PUPPET_GRIPPER_JOINT_CLOSE) / (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE))

MASTER_GRIPPER_JOINT_MID = (MASTER_GRIPPER_JOINT_OPEN + MASTER_GRIPPER_JOINT_CLOSE)/2
