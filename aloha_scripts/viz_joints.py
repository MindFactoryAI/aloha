import rospy
from sensor_msgs.msg import JointState
from argparse import ArgumentParser
from functools import partial
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from interbotix_xs_msgs.srv import RegisterValues


class SensorQueue:
    def __init__(self):
        self.pos = deque([0] * 800, maxlen=800)
        self.effort = deque([0] * 800, maxlen=800)
        self.present_current = deque([0] * 800, maxlen=800)
        self.goal_current = deque([0] * 800, maxlen=800)
        self.current_limit = deque([0] * 800, maxlen=800)
        self.operating_mode = deque([0] * 800, maxlen=800)


qq = {
    'master_right': SensorQueue(),
    'puppet_right': SensorQueue(),
    'master_left': SensorQueue(),
    'puppet_left': SensorQueue(),
}

fig, [
    [left_pos_ax, right_pos_ax],
    [left_effort_ax, right_effort_ax],
    [left_current_ax, right_current_ax],
    [left_mode_ax, right_mode_ax]
] = plt.subplots(4, 2, figsize=(24, 12), dpi=80)

current_plot = {}
goal_current_plot = {}
current_limit_plot = {}
operating_mode_plot = {}

props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
textstr = f'0 Current Control Mode\n' \
          f'1 Velocity Control Mode\n' \
          f'3(Default) Position Control Mode\n' \
          f'4 Extended Position Control Mode(Multi-turn) \n' \
          f'5 Current-based Position Control Mode\n' \
          f'16 PWM Control Mode (Voltage Control Mode)'

###########
# Right arm
###########

# position plot
master_right_plot, = right_pos_ax.plot(qq['master_right'].pos, label='master_position')
puppet_right_plot, = right_pos_ax.plot(qq['puppet_right'].pos, label='puppet_position')
right_pos_ax.set_ylim(-1, 1.5)
right_pos_ax.legend()

# effort plot
puppet_right_effort_plot, = right_effort_ax.plot(qq['puppet_right'].effort, label='effort')
puppet_right_effort_max_plot, = right_effort_ax.plot([max(qq['puppet_right'].effort)]*len(qq['puppet_right'].effort), label='max_effort')
right_effort_ax.set_ylim(0, 1200)
right_effort_ax.legend()

# current ax
current_plot['puppet_right'], = right_current_ax.plot(qq['puppet_right'].present_current, label='present_current')
goal_current_plot['puppet_right'], = right_current_ax.plot(qq['puppet_right'].goal_current, label='goal_current')
current_limit_plot['puppet_right'], = right_current_ax.plot(qq['puppet_right'].current_limit, label='current_limit')
right_current_ax.set_ylim(-1200, 1200)
right_current_ax.legend()

# mode plot
operating_mode_plot['puppet_right'], = right_mode_ax.plot(qq['puppet_right'].operating_mode, label='operating_mode')
right_mode_ax.set_ylim(0, 6)
right_mode_ax.legend()


##########
# Left arm
##########

# position plot
master_left_plot, = left_pos_ax.plot(qq['master_left'].pos, label='master_position')
puppet_left_plot, = left_pos_ax.plot(qq['puppet_left'].pos, label='puppet_position')
left_pos_ax.set_ylim(-1, 1.5)
left_pos_ax.legend()

# effort plot
puppet_left_effort_plot, = left_effort_ax.plot(qq['puppet_left'].effort, label='effort')
puppet_left_effort_max_plot, = left_effort_ax.plot([max(qq['puppet_left'].effort)]*len(qq['puppet_left'].effort), label='max_effort')
left_effort_ax.set_ylim(0, 1200)
left_effort_ax.legend()

# current plot
current_plot['puppet_left'], = left_current_ax.plot(qq['puppet_left'].present_current, label='present_current')
goal_current_plot['puppet_left'], = left_current_ax.plot(qq['puppet_left'].goal_current, label='goal_current')
current_limit_plot['puppet_left'], = left_current_ax.plot(qq['puppet_left'].current_limit, label='current_limit')
left_current_ax.set_ylim(-1200, 1200)
left_current_ax.legend()

# operating mode plot
operating_mode_plot['puppet_left'], = left_mode_ax.plot(qq['puppet_left'].operating_mode, label='operating_mode')
left_mode_ax.set_ylim(0, 6)
left_mode_ax.text(0.05, 0.95, textstr, transform=left_mode_ax.transAxes, fontsize=8,
        verticalalignment='top', bbox=props)
left_mode_ax.legend()


def find_first_match_index(strings, target):
    for index, string in enumerate(strings):
        if string == target:
            return index
    return -1  # Return -1 if no match is found


global step
step = 0


def joint_states_callback(arm, args, msg):
    global step
    i = find_first_match_index(msg.name, 'gripper')

    if step % 4 == 0:

        # Access the joint state values
        names = msg.name[i]
        velocities = msg.velocity[i]

        qq[arm].pos.append(msg.position[i])
        qq[arm].effort.append(msg.effort[i])

    step += 1
    # Print the joint state values
    # print("Topic: {}".format(arm))
    # print("Joint Name: {}".format(names))
    # print("Joint Positions: {}".format(pos))
    # # print("Joint Velocities: {}".format(velocities))
    # print("Joint Efforts: {}".format(effort))


def update_plot(frame):

    for arm in ['puppet_left', 'puppet_right']:

        current = service_command(f'/{arm}/get_motor_registers', 'single', 'gripper', 'Present_Current')
        qq[arm].present_current.append(current)
        current_plot[arm].set_ydata(qq[arm].present_current)

        goal_current = service_command(f'/{arm}/get_motor_registers', 'single', 'gripper', 'Goal_Current')
        qq[arm].goal_current.append(goal_current)
        goal_current_plot[arm].set_ydata(qq[arm].goal_current)

        current_limit = service_command(f'/{arm}/get_motor_registers', 'single', 'gripper', 'Current_Limit')
        qq[arm].current_limit.append(current_limit)
        current_limit_plot[arm].set_ydata(qq[arm].current_limit)

        operating_mode = service_command(f'/{arm}/get_motor_registers', 'single', 'gripper', 'Operating_Mode')
        qq[arm].operating_mode.append(operating_mode)
        operating_mode_plot[arm].set_ydata(qq[arm].operating_mode)

    master_right_plot.set_ydata(qq['master_right'].pos)
    puppet_right_plot.set_ydata(qq['puppet_right'].pos)
    master_left_plot.set_ydata(qq['master_left'].pos)
    puppet_left_plot.set_ydata(qq['puppet_left'].pos)
    puppet_right_effort_plot.set_ydata(qq['puppet_right'].effort)
    puppet_left_effort_plot.set_ydata(qq['puppet_left'].effort)
    puppet_right_effort_max_plot.set_ydata([max(qq['puppet_right'].effort)]*len(qq['puppet_right'].effort))
    puppet_left_effort_max_plot.set_ydata([max(qq['puppet_left'].effort)]*len(qq['puppet_left'].effort))


def service_command(service_path, cmd_type, name, reg, value=0):
    rospy.wait_for_service(service_path)

    try:
        get_registers = rospy.ServiceProxy(service_path, RegisterValues)

        # Make the service call
        response = get_registers(cmd_type, name, reg, value)

        # Process the response
        if response:
            # print("Register values: ", response.values)
            return int(response.values[0])
        else:
            # print("Failed to get register values.")
            return 0

    except rospy.ServiceException as e:
        # print("Service call failed:", str(e))
        return 0


def main(args):

    # Initialize the ROS node
    rospy.init_node('joint_states_subscriber')

    master_right_joint_states_cb_partial = partial(joint_states_callback, 'master_right', args)
    puppet_right_joint_states_cb_partial = partial(joint_states_callback, 'puppet_right', args)
    master_left_joint_states_cb_partial = partial(joint_states_callback, 'master_left', args)
    puppet_left_joint_states_cb_partial = partial(joint_states_callback, 'puppet_left', args)

    # Subscribe to the joint states topic
    rospy.Subscriber('/master_right/joint_states', JointState, master_right_joint_states_cb_partial)
    rospy.Subscriber('/puppet_right/joint_states', JointState, puppet_right_joint_states_cb_partial)
    rospy.Subscriber('/master_left/joint_states', JointState, master_left_joint_states_cb_partial)
    rospy.Subscriber('/puppet_left/joint_states', JointState, puppet_left_joint_states_cb_partial)

    # ani = FuncAnimation(fig, update_plot, fargs=(master_right_plot, puppet_right_plot, master_left_plot, puppet_left_plot, ), interval=100)
    ani = FuncAnimation(fig, update_plot, interval=10)
    plt.show()
    # Spin ROS event loop
    rospy.spin()


if __name__ == '__main__':
    parser = ArgumentParser()
    # parser.add_argument('joint_name')
    args = parser.parse_args()
    main(args)
