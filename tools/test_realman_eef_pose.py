#!/usr/bin/env python3

import numpy as np

from lerobot_robot_realman.config_realman import RealmanConfig
from lerobot_robot_realman.realman import RealmanRobot


def main():
    cfg = RealmanConfig(
        control_mode="eef_pose",
        enable_motion=True,
        enable_camera_observation=False,
        use_gripper=True,
        enable_gripper=True,
    )
    robot = RealmanRobot(cfg)
    robot.connect()

    try:
        obs = robot.get_observation()
        current_pose = np.asarray(obs["observation.eef_pose"], dtype=np.float32)
        print("current joint/state:", obs["observation.state"])
        print("current eef_pose:", current_pose)

        target_pose = current_pose.copy()
        target_pose[0] += 0.02
        action = np.concatenate([target_pose, np.array([1.0], dtype=np.float32)], axis=0)
        print("sending eef action:", action.tolist())
        ret = robot.send_action({"action": action})
        print("returned action:", ret["action"])
    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()
