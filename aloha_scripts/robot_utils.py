import numpy as np
import time
import sys
import termios
import tty

from aloha_scripts.constants import DT, MASTER_GRIPPER_JOINT_MID, PUPPET2MASTER_JOINT_FN
from interbotix_xs_msgs.msg import JointSingleCommand
from interbotix_xs_msgs.srv import RegisterValues
import rospy

import IPython

e = IPython.embed


class ImageRecorder:
    def __init__(self, init_node=True, is_debug=False):
        from collections import deque
        import rospy
        from cv_bridge import CvBridge
        from sensor_msgs.msg import CompressedImage
        self.is_debug = is_debug
        self.bridge = CvBridge()
        self.camera_names = ['cam_high', 'cam_low', 'cam_left_wrist', 'cam_right_wrist']
        if init_node:
            rospy.init_node('image_recorder', anonymous=True)
        for cam_name in self.camera_names:
            setattr(self, f'{cam_name}_image_compressed', None)
            setattr(self, f'{cam_name}_image', None)
            setattr(self, f'{cam_name}_secs', None)
            setattr(self, f'{cam_name}_nsecs', None)
            if cam_name == 'cam_high':
                callback_func = self.image_cb_cam_high
            elif cam_name == 'cam_low':
                callback_func = self.image_cb_cam_low
            elif cam_name == 'cam_left_wrist':
                callback_func = self.image_cb_cam_left_wrist
            elif cam_name == 'cam_right_wrist':
                callback_func = self.image_cb_cam_right_wrist
            else:
                raise NotImplementedError
            rospy.Subscriber(f"/usb_{cam_name}/image_raw/compressed", CompressedImage, callback_func)
            if self.is_debug:
                setattr(self, f'{cam_name}_timestamps', deque(maxlen=50))
        time.sleep(0.5)

    def image_cb(self, cam_name, data):
        setattr(self, f'{cam_name}_image_compressed', np.frombuffer(data.data, np.uint8))
        setattr(self, f'{cam_name}_image', self.bridge.compressed_imgmsg_to_cv2(data, desired_encoding='passthrough'))
        setattr(self, f'{cam_name}_secs', data.header.stamp.secs)
        setattr(self, f'{cam_name}_nsecs', data.header.stamp.nsecs)
        # cv2.imwrite('/home/tonyzhao/Desktop/sample.jpg', cv_image)
        if self.is_debug:
            getattr(self, f'{cam_name}_timestamps').append(data.header.stamp.secs + data.header.stamp.secs * 1e-9)

    def image_cb_cam_high(self, data):
        cam_name = 'cam_high'
        return self.image_cb(cam_name, data)

    def image_cb_cam_low(self, data):
        cam_name = 'cam_low'
        return self.image_cb(cam_name, data)

    def image_cb_cam_left_wrist(self, data):
        cam_name = 'cam_left_wrist'
        return self.image_cb(cam_name, data)

    def image_cb_cam_right_wrist(self, data):
        cam_name = 'cam_right_wrist'
        return self.image_cb(cam_name, data)

    def get_images(self):
        image_dict = dict()
        image_dict_compressed = dict()
        for cam_name in self.camera_names:
            image_dict[cam_name] = getattr(self, f'{cam_name}_image')
            image_dict_compressed[cam_name] = getattr(self, f'{cam_name}_image_compressed')
        return image_dict, image_dict_compressed

    def print_diagnostics(self):
        def dt_helper(l):
            l = np.array(l)
            diff = l[1:] - l[:-1]
            return np.mean(diff)

        for cam_name in self.camera_names:
            image_freq = 1 / dt_helper(getattr(self, f'{cam_name}_timestamps'))
            print(f'{cam_name} {image_freq=:.2f}')
        print()


class Recorder:
    def __init__(self, side, init_node=True, is_debug=False):
        from collections import deque
        import rospy
        from sensor_msgs.msg import JointState
        from interbotix_xs_msgs.msg import JointGroupCommand, JointSingleCommand

        self.secs = None
        self.nsecs = None
        self.qpos = None
        self.effort = None
        self.arm_command = None
        self.gripper_command = None
        self.is_debug = is_debug

        if init_node:
            rospy.init_node('recorder', anonymous=True)
        rospy.Subscriber(f"/puppet_{side}/joint_states", JointState, self.puppet_state_cb)
        rospy.Subscriber(f"/puppet_{side}/commands/joint_group", JointGroupCommand, self.puppet_arm_commands_cb)
        rospy.Subscriber(f"/puppet_{side}/commands/joint_single", JointSingleCommand, self.puppet_gripper_commands_cb)
        if self.is_debug:
            self.joint_timestamps = deque(maxlen=50)
            self.arm_command_timestamps = deque(maxlen=50)
            self.gripper_command_timestamps = deque(maxlen=50)
        time.sleep(0.1)

    def puppet_state_cb(self, data):
        self.qpos = data.position
        self.qvel = data.velocity
        self.effort = data.effort
        self.data = data
        if self.is_debug:
            self.joint_timestamps.append(time.time())

    def puppet_arm_commands_cb(self, data):
        self.arm_command = data.cmd
        if self.is_debug:
            self.arm_command_timestamps.append(time.time())

    def puppet_gripper_commands_cb(self, data):
        self.gripper_command = data.cmd
        if self.is_debug:
            self.gripper_command_timestamps.append(time.time())

    def print_diagnostics(self):
        def dt_helper(l):
            l = np.array(l)
            diff = l[1:] - l[:-1]
            return np.mean(diff)

        joint_freq = 1 / dt_helper(self.joint_timestamps)
        arm_command_freq = 1 / dt_helper(self.arm_command_timestamps)
        gripper_command_freq = 1 / dt_helper(self.gripper_command_timestamps)

        print(f'{joint_freq=:.2f}\n{arm_command_freq=:.2f}\n{gripper_command_freq=:.2f}\n')


def get_arm_joint_positions(bot):
    return bot.arm.core.joint_states.position[:6]


def get_arm_gripper_positions(bot):
    joint_position = bot.gripper.core.joint_states.position[6]
    return joint_position


def move_arms(bot_list, target_pose_list, move_time=1):
    num_steps = int(move_time / DT)
    curr_pose_list = [get_arm_joint_positions(bot) for bot in bot_list]
    traj_list = [np.linspace(curr_pose, target_pose, num_steps) for curr_pose, target_pose in
                 zip(curr_pose_list, target_pose_list)]
    for t in range(num_steps):
        for bot_id, bot in enumerate(bot_list):
            bot.arm.set_joint_positions(traj_list[bot_id][t], blocking=False)
        time.sleep(DT)


def move_grippers(bot_list, target_pose_list, move_time):
    gripper_command = JointSingleCommand(name="gripper")
    num_steps = int(move_time / DT)
    curr_pose_list = [get_arm_gripper_positions(bot) for bot in bot_list]
    traj_list = [np.linspace(curr_pose, target_pose, num_steps) for curr_pose, target_pose in
                 zip(curr_pose_list, target_pose_list)]
    for t in range(num_steps):
        for bot_id, bot in enumerate(bot_list):
            gripper_command.cmd = traj_list[bot_id][t]
            bot.gripper.core.pub_single.publish(gripper_command)
        time.sleep(DT)


def setup_puppet_bot(bot):
    bot.dxl.robot_reboot_motors("single", "gripper", True)
    bot.dxl.robot_set_operating_modes("group", "arm", "position")
    bot.dxl.robot_set_operating_modes("single", "gripper", "current_based_position")
    torque_on(bot)


def setup_master_bot(bot):
    bot.dxl.robot_set_operating_modes("group", "arm", "pwm")
    bot.dxl.robot_set_operating_modes("single", "gripper", "current_based_position")
    torque_off(bot)


def set_standard_pid_gains(bot):
    bot.dxl.robot_set_motor_registers("group", "arm", 'Position_P_Gain', 800)
    bot.dxl.robot_set_motor_registers("group", "arm", 'Position_I_Gain', 0)


def set_low_pid_gains(bot):
    bot.dxl.robot_set_motor_registers("group", "arm", 'Position_P_Gain', 100)
    bot.dxl.robot_set_motor_registers("group", "arm", 'Position_I_Gain', 0)


def torque_off(bot):
    bot.dxl.robot_torque_enable("group", "arm", False)
    bot.dxl.robot_torque_enable("single", "gripper", False)


def torque_on(bot):
    bot.dxl.robot_torque_enable("group", "arm", True)
    bot.dxl.robot_torque_enable("single", "gripper", True)


def gripper_torque_on(bot):
    bot.dxl.robot_torque_enable("single", "gripper", True)


def gripper_torque_off(bot):
    bot.dxl.robot_torque_enable("single", "gripper", False)


def service_command(service_path, cmd_type, name, reg, value=0):
    rospy.wait_for_service(service_path)

    try:
        get_registers = rospy.ServiceProxy(service_path, RegisterValues)

        # Make the service call
        response = get_registers(cmd_type, name, reg, value)

        # Process the response
        if response:
            return str(response.values)
        else:
            return "Failed"

    except rospy.ServiceException as e:
        print("Service call failed:", str(e))
        return "Failed"


def reboot_grippers(master_bot_left, master_bot_right, puppet_bot_left, puppet_bot_right, current_limit=1000):
    for bot in [master_bot_left, master_bot_right]:
        bot.dxl.robot_reboot_motors("single", "gripper", True)

    for bot in [puppet_bot_left, puppet_bot_right]:
        bot.dxl.robot_reboot_motors("single", "gripper", True)
        bot.dxl.robot_torque_enable("single", "gripper", False)
        bot.dxl.robot_set_motor_registers("single", "gripper", 'Current_Limit', current_limit)
        bot.dxl.robot_set_operating_modes("single", "gripper", "current_based_position")
        bot.dxl.robot_torque_enable("single", "gripper", True)


def reboot_arms(master_bot_left, master_bot_right, puppet_bot_left, puppet_bot_right, current_limit=1000):
    """ Move all 4 robots to a pose where it is easy to start demonstration """

    # reboot gripper motors, and set operating modes for all motors

    reboot_grippers(master_bot_left, master_bot_right, puppet_bot_left, puppet_bot_right, current_limit)

    for bot in [puppet_bot_left, puppet_bot_right]:
        bot.dxl.robot_set_operating_modes("group", "arm", "position")

    for bot in [master_bot_left, master_bot_right]:
        bot.dxl.robot_set_operating_modes("group", "arm", "position")
        bot.dxl.robot_set_operating_modes("single", "gripper", "position")


def getch():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        char = sys.stdin.read(1)
        if char == '\x1b':
            # If the first character is an escape sequence, read more characters
            char += sys.stdin.read(2)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return char


def wait_for_input(env, master_bot_left, master_bot_right, close_thresh=0.25, block_until="double_close",
                   message='Close the gripper to start'):
    """
    Sets the master handles to center, and waits for user to close or open the handles
    after user presses, it will move handles to puppet grippers current position
    master_gripper torque will remain on after this is called

    note: the puppet_gripper will not be effected by this call

    close_thresh: the amount the user must move the handles in joint space to trigger
    block_until:
        "any": will block until either handle is closed or open
        "double": requires both handles to be in either a closed or open state
        "double_close": requires both handles to close before proceeding

    returns np.array([left_state, right_state]) where state: -1: closed, 0: middle, +1 open
    """

    master_bot_left.dxl.robot_torque_enable("single", "gripper", True)
    master_bot_right.dxl.robot_torque_enable("single", "gripper", True)

    move_grippers([master_bot_left, master_bot_right], [MASTER_GRIPPER_JOINT_MID, MASTER_GRIPPER_JOINT_MID], 0.2)

    # press gripper to start data collection
    # disable torque for only gripper joint of master robot to allow user movement
    master_bot_left.dxl.robot_torque_enable("single", "gripper", False)
    master_bot_right.dxl.robot_torque_enable("single", "gripper", False)
    print(message)

    if block_until == 'keyboard':
        ch = getch()
        print(ch)
        return ch

    # left_closed, right_closed, left_opened, right_opened = False, False, False, False
    while True:

        d_left = get_arm_gripper_positions(master_bot_left) - MASTER_GRIPPER_JOINT_MID
        d_right = get_arm_gripper_positions(master_bot_right) - MASTER_GRIPPER_JOINT_MID
        delta_normalized = np.array([d_left, d_right]) / close_thresh
        handle_state = np.where(np.abs(delta_normalized) < 1., 0, np.sign(delta_normalized))

        if block_until == "double_close":
            if BOTH_CLOSED(handle_state):
                break
        elif block_until == "double":
            if BOTH_CLOSED(handle_state) or BOTH_OPEN(handle_state):
                break
        elif block_until == "any":
            if ANY_CLOSED_OR_OPEN(handle_state):
                break
        else:
            raise Exception("wait_for_input has invalid block_until mode: valid modes are double_close, double, any")

        time.sleep(DT / 10)

    master_bot_left.dxl.robot_torque_enable("single", "gripper", True)
    master_bot_right.dxl.robot_torque_enable("single", "gripper", True)

    master_bot_left_pos = PUPPET2MASTER_JOINT_FN(get_arm_gripper_positions(env.puppet_bot_left))
    master_bot_right_pos = PUPPET2MASTER_JOINT_FN(get_arm_gripper_positions(env.puppet_bot_right))
    move_grippers([master_bot_left, master_bot_right], [master_bot_left_pos, master_bot_right_pos], 0.2)
    print(f'left: {handle_state[0]}, right: {handle_state[1]}')
    return handle_state


ANY_CLOSED_OR_OPEN = lambda handle_state: np.any(handle_state != 0)
LEFT_HANDLE_CLOSED = lambda handle_state: handle_state[0] == -1.
RIGHT_HANDLE_CLOSED = lambda handle_state: handle_state[1] == -1.
LEFT_HANDLE_OPEN = lambda handle_state: handle_state[0] == 1.
RIGHT_HANDLE_OPEN = lambda handle_state: handle_state[1] == 1.
BOTH_CLOSED = lambda handle_state: np.all(handle_state == -1.)
BOTH_OPEN = lambda handle_state: np.all(handle_state == 1.)

