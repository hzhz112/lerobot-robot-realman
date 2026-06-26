#!/usr/bin/env python3

import numpy as np

from lerobot_robot_realman.config_realman import RealmanConfig
from lerobot_robot_realman.realman import RealmanRobot


def main():
    cfg = RealmanConfig(
        control_mode="joint",
        enable_motion=True,
        enable_camera_observation=False,
        use_gripper=True,
        enable_gripper=True,
    )
    robot = RealmanRobot(cfg)
    robot.connect()

    try:
        obs = robot.get_observation()
        print("current joint/state:", obs["observation.state"])
        print("current eef_pose:", obs["observation.eef_pose"])

        action = np.array([0.0, 0.0, 90.0, 0.0, 90.0, 0.0, 1.0], dtype=np.float32)
        print("sending joint action:", action.tolist())
        ret = robot.send_action({"action": action})
        print("returned action:", ret["action"])
    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()
