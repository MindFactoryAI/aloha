#!/usr/bin/env python

import rospy
from interbotix_xs_msgs.srv import RegisterValues


def set_gripper_register():
    rospy.wait_for_service('/puppet_left/set_motor_registers')

    try:
        get_registers = rospy.ServiceProxy('/puppet_left/set_motor_registers', RegisterValues)

        # Specify the arguments for the service call
        cmd_type = 'single'  # Replace with the appropriate command type
        name = 'gripper'
        reg = 'Current_Limit'
        value = 300

        # Make the service call
        response = get_registers(cmd_type, name, reg, value)

        # Process the response
        if response:
            print("Register values: ", response.values)
        else:
            print("Failed to get register values.")

    except rospy.ServiceException as e:
        print("Service call failed:", str(e))


if __name__ == '__main__':
    rospy.init_node('gripper_register_setter_node')
    set_gripper_register()