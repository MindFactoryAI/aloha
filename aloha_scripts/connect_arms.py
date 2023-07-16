from interbotix_xs_modules.arm import InterbotixManipulatorXS

puppet_bot_right = InterbotixManipulatorXS(robot_model="vx300s", group_name="arm", gripper_name="gripper", robot_name=f'puppet_right', init_node=True)

print(puppet_bot_right)