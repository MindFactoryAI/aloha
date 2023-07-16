#!/usr/bin/env python

import rospy
from interbotix_xs_msgs.srv import RegisterValues


def service_command(service_path, cmd_type, name, reg, value=0):
    rospy.wait_for_service(service_path)

    try:
        get_registers = rospy.ServiceProxy(service_path, RegisterValues)

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
    rospy.init_node('gripper_register_node')
    service_command('/puppet_left/get_motor_registers', 'single', 'gripper', 'Current_Limit')
    service_command('/puppet_left/set_motor_registers', 'single', 'gripper', 'Current_Limit', 800)
    service_command('/puppet_left/get_motor_registers', 'single', 'gripper', 'Current_Limit')

    service_command('/puppet_left/get_motor_registers', 'single', 'gripper', 'Present_Current')
