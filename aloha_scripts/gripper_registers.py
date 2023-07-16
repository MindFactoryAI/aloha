#!/usr/bin/env python

import rospy
from interbotix_xs_msgs.srv import RegisterValues
from rich.table import Table
from rich.console import Console


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


if __name__ == '__main__':

    for gripper in ['puppet_left', 'puppet_right']:
        ros_service = f'/{gripper}/get_motor_registers'
        table = Table(show_header=True, header_style="bold magenta", title=f'[yellow]{ros_service}')
        table.add_column("Register", style="cyan", no_wrap=True)
        table.add_column("Value", style="cyan")

        for register_name in ['Operating_Mode', 'Goal_Current', 'Current_Limit', 'Hardware_Error_Status']:
            value = service_command(ros_service, 'single', 'gripper', f'{register_name}')
            table.add_row(register_name, value)

        console = Console()
        console.print(table)