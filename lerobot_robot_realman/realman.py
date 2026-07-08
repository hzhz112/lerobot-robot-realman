from typing import Any
import time
import numpy as np
from Robotic_Arm.rm_robot_interface import RoboticArm, rm_thread_mode_e

from lerobot.cameras import make_cameras_from_configs
from lerobot.robots.robot import Robot

from .config_realman import RealmanConfig
from .gripper import HitbotGripper


class RealmanRobot(Robot):
    config_class = RealmanConfig
    name = "realman"

    def __init__(self, config: RealmanConfig):
        super().__init__(config)
        self.config = config
        self.arm = None
        self.handle = None
        self._connected = False
        self.cameras = make_cameras_from_configs(config.cameras)
        self.gripper = HitbotGripper(
            port=config.gripper_port,
            enable=config.enable_gripper,
            deadband=config.gripper_cmd_deadband,
        )
        self._last_joint_deg = None
        self._last_pose = None
        self._teleop_initial_pose = None

    def _joint_state_keys(self) -> list[str]:
        keys = [f"joint_{i + 1}" for i in range(self.config.dof)]
        if self.config.use_gripper:
            keys.append("gripper")
        return keys

    def _eef_pose_keys(self) -> list[str]:
        return ["eef_x", "eef_y", "eef_z", "eef_rx", "eef_ry", "eef_rz"]

    def _action_keys(self) -> list[str]:
        if self.config.control_mode == "joint":
            return self._joint_state_keys()

        keys = ["x", "y", "z", "rx", "ry", "rz"]
        if self.config.use_dummy_action and self.config.use_gripper:
            keys.append("dummy")
        if self.config.use_gripper:
            keys.append("gripper")
        return keys

    @property
    def observation_features(self) -> dict:
        features = {key: float for key in [*self._joint_state_keys(), *self._eef_pose_keys()]}
        for cam_name, cam_cfg in self.config.cameras.items():
            features[cam_name] = (
                cam_cfg.height,
                cam_cfg.width,
                3,
            )
        return features

    @property
    def action_features(self) -> dict:
        return {key: float for key in self._action_keys()}

    @property
    def is_connected(self) -> bool:
        if not self._connected:
            return False
        for cam in self.cameras.values():
            if not cam.is_connected:
                return False
        return True

    @property
    def is_calibrated(self) -> bool:
        return True

    def connect(self, calibrate: bool = True) -> None:
        if self._connected:
            print("[RealmanRobot] already connected.")
            return

        self.arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
        self.handle = self.arm.rm_create_robot_arm(
            self.config.ip,
            self.config.port,
            self.config.connection_level,
        )
        print(f"[RM65] connected to {self.config.ip}:{self.config.port}")
        print(f"[RM65] handle: {self.handle}")

        if getattr(self.handle, "id", -1) == -1:
            raise RuntimeError(
                f"Failed to create robot arm handle for {self.config.ip}:{self.config.port} "
                f"(connection_level={self.config.connection_level})"
            )

        try:
            timeout_ret = self.arm.rm_set_timeout(self.config.sdk_timeout_ms)
            print(f"[RM65] rm_set_timeout({self.config.sdk_timeout_ms}) ret: {timeout_ret}")
        except Exception as e:
            print(f"[RM65] rm_set_timeout failed: {e}")

        if self.config.enable_movev_canfd_init:
            try:
                movev_ret = self.arm.rm_set_movev_canfd_init(
                    self.config.movev_canfd_follow,
                    self.config.movev_canfd_frame_type,
                    self.config.movev_canfd_dt_ms,
                )
                print(
                    "[RM65] rm_set_movev_canfd_init"
                    f"({self.config.movev_canfd_follow}, "
                    f"{self.config.movev_canfd_frame_type}, "
                    f"{self.config.movev_canfd_dt_ms}) ret: {movev_ret}"
                )
            except Exception as e:
                print(f"[RM65] rm_set_movev_canfd_init failed: {e}")

        state_code, state = self.arm.rm_get_current_arm_state()
        if state_code != 0:
            raise RuntimeError(
                f"RM65 connection established but rm_get_current_arm_state failed: code={state_code}, state={state}"
            )
        print(f"[RM65] current arm state verified.")

        try:
            info_code, info = self.arm.rm_get_robot_info()
            if info_code == 0:
                print(f"[RM65] robot info: {info}")
        except Exception as e:
            print(f"[RM65] rm_get_robot_info failed: {e}")

        code, joint = self.arm.rm_get_joint_degree()
        if code != 0:
            raise RuntimeError(f"RM65 connection failed. rm_get_joint_degree code={code}, joint={joint}")
        print(f"[RM65] initial joint: {joint}")

        for cam_name, cam in self.cameras.items():
            cam.connect()
            print(f"[Camera] connected: {cam_name}")

        self.gripper.connect()

        if self.config.reset_on_connect and self.config.enable_motion:
            print(f"[RM65] resetting to initial joints: {self.config.initial_joint6}")
            self.reset_to_initial()
            time.sleep(0.5)

        self._last_joint_deg = self._read_joint_deg()
        self._last_pose = self._read_eef_pose()
        self._teleop_initial_pose = self._last_pose.copy()
        self._connected = True

        if self.config.enable_motion:
            print("[WARNING] enable_motion=True，机械臂会真实运动。")
        else:
            print("[SAFE MODE] enable_motion=False，只打印动作，不下发机械臂。")

    def disconnect(self) -> None:
        for cam_name, cam in self.cameras.items():
            try:
                if cam.is_connected:
                    cam.disconnect()
                    print(f"[Camera] disconnected: {cam_name}")
            except Exception as e:
                print(f"[Camera] disconnect error {cam_name}: {e}")

        if self.config.use_gripper:
            self.gripper.close_device()

        if self.arm is not None and self.config.reset_on_disconnect:
            try:
                self.reset_to_initial()
            except Exception as e:
                print(f"[RM65 RESET ERROR] {e}")

        if self.arm is not None:
            try:
                self.arm.rm_delete_robot_arm()
                print("[RM65] disconnected.")
            except Exception as e:
                print(f"[RM65] disconnect error: {e}")

        self.arm = None
        self.handle = None
        self._teleop_initial_pose = None
        self._connected = False

    def configure(self) -> None:
        if self.arm is not None:
            try:
                self._teleop_initial_pose = self._read_eef_pose()
            except Exception:
                self._teleop_initial_pose = None

    def calibrate(self) -> None:
        pass

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise ConnectionError("RealmanRobot is not connected.")

        joint6 = self._read_joint_deg().astype(np.float32)
        pose6 = self._read_eef_pose().astype(np.float32)
        gripper_value = self._read_gripper()

        if self.config.use_gripper:
            state = np.concatenate([joint6, np.array([gripper_value], dtype=np.float32)], axis=0)
        else:
            state = joint6

        obs: dict[str, Any] = {
            **{key: float(value) for key, value in zip(self._joint_state_keys(), state.tolist(), strict=True)},
            **{key: float(value) for key, value in zip(self._eef_pose_keys(), pose6.tolist(), strict=True)},
        }

        if not self.config.enable_camera_observation:
            return obs

        for cam_name, cam in self.cameras.items():
            try:
                image = cam.async_read()
            except Exception:
                image = cam.read()
            obs[cam_name] = image

        return obs

    def send_action(self, action: dict[str, Any] | np.ndarray) -> dict[str, Any]: #获取action的信息 然后下发给机械臂
        if not self.is_connected:
            raise ConnectionError("RealmanRobot is not connected.")

        if isinstance(action, dict):
            if "action" in action:
                target = np.asarray(action["action"], dtype=np.float32).reshape(-1)
            else:
                target = np.asarray([action[key] for key in self._action_keys()], dtype=np.float32).reshape(-1)
        else:
            target = np.asarray(action, dtype=np.float32).reshape(-1)

        if self.config.control_mode == "joint":
            sent = self._send_joint_action(target)
        elif self.config.control_mode == "eef_pose":
            sent = self._send_pose_action(target)
        elif self.config.control_mode == "eef_velocity":
            sent = self._send_pose_velocity_action(target)
        elif self.config.control_mode == "delta_eef":
            sent = self._send_delta_pose_action(target)
        else:
            raise ValueError(f"Unknown control_mode: {self.config.control_mode}")

        return {key: float(value) for key, value in zip(self._action_keys(), sent.tolist(), strict=True)}

    def _send_joint_action(self, target: np.ndarray) -> np.ndarray:
        expected_min = self.config.dof
        if target.shape[0] < expected_min:
            raise ValueError(f"joint action dim should >= {expected_min}, got {target.shape}")

        joint_cmd = target[: self.config.dof]
        gripper_cmd = None
        if self.config.use_gripper and target.shape[0] >= self.config.dof + 1:
            gripper_cmd = float(target[self.config.dof])

        joint_safe = self._clip_joint_action(joint_cmd)
        if gripper_cmd is not None:
            self._send_gripper(gripper_cmd)

        if self.config.enable_motion:
            self._send_joint_deg(joint_safe)
        self._last_joint_deg = joint_safe.copy()

        if gripper_cmd is not None:
            return np.concatenate([joint_safe, np.array([gripper_cmd], dtype=np.float32)], axis=0)
        return joint_safe

    def _send_pose_action(self, target: np.ndarray) -> np.ndarray:
        if target.shape[0] < 6:
            raise ValueError(f"pose action dim should >= 6, got {target.shape}")

        pose_raw = target[:6]
        gripper_cmd = None
        if self.config.use_gripper:
            if self.config.use_dummy_action:
                if target.shape[0] < 8:
                    raise ValueError(f"pose8 action expected dim >= 8, got {target.shape}")
                gripper_cmd = float(target[7])
            elif target.shape[0] >= 7:
                gripper_cmd = float(target[6])

        current_pose = self._read_eef_pose()
        pose_safe = self._safety_filter_pose(pose_raw, current_pose)
        if gripper_cmd is not None:
            self._send_gripper(gripper_cmd)
        if self.config.enable_motion:
            self._send_eef_pose(pose_safe)
        self._last_pose = pose_safe.copy()

        if gripper_cmd is not None:
            if self.config.use_dummy_action:
                return np.concatenate([pose_safe, np.array([0.0], dtype=np.float32), np.array([gripper_cmd], dtype=np.float32)], axis=0)
            return np.concatenate([pose_safe, np.array([gripper_cmd], dtype=np.float32)], axis=0)
        return pose_safe

    def _send_pose_velocity_action(self, target: np.ndarray) -> np.ndarray:
        if target.shape[0] < 6:
            raise ValueError(f"pose action dim should >= 6, got {target.shape}")

        pose_raw = target[:6]
        gripper_cmd = None
        if self.config.use_gripper:
            if self.config.use_dummy_action:
                if target.shape[0] < 8:
                    raise ValueError(f"pose8 action expected dim >= 8, got {target.shape}")
                gripper_cmd = float(target[7])
            elif target.shape[0] >= 7:
                gripper_cmd = float(target[6])

        current_pose = self._read_eef_pose()
        pose_safe = self._safety_filter_pose(pose_raw, current_pose)
        if gripper_cmd is not None:
            self._send_gripper(gripper_cmd)
        if self.config.enable_motion:
            self._send_eef_velocity_towards(current_pose, pose_safe)
        self._last_pose = pose_safe.copy()

        if gripper_cmd is not None:
            if self.config.use_dummy_action:
                return np.concatenate([pose_safe, np.array([0.0], dtype=np.float32), np.array([gripper_cmd], dtype=np.float32)], axis=0)
            return np.concatenate([pose_safe, np.array([gripper_cmd], dtype=np.float32)], axis=0)
        return pose_safe

    def _send_delta_pose_action(self, target: np.ndarray) -> np.ndarray:
        if target.shape[0] < 6:
            raise ValueError(f"delta pose action dim should >= 6, got {target.shape}")

        delta_raw = np.asarray(target[:6], dtype=np.float32)
        gripper_cmd = None
        if self.config.use_gripper:
            if self.config.use_dummy_action:
                if target.shape[0] < 8:
                    raise ValueError(f"delta pose8 action expected dim >= 8, got {target.shape}")
                gripper_cmd = float(target[7])
            elif target.shape[0] >= 7:
                gripper_cmd = float(target[6])

        pos_order = list(self.config.delta_eef_position_axis_order)
        pos_sign = np.asarray(self.config.delta_eef_position_axis_sign, dtype=np.float32)
        pos_scale = np.asarray(self.config.delta_eef_position_scale, dtype=np.float32)
        rot_order = list(self.config.delta_eef_rotation_axis_order)
        rot_sign = np.asarray(self.config.delta_eef_rotation_axis_sign, dtype=np.float32)
        rot_scale = np.asarray(self.config.delta_eef_rotation_scale, dtype=np.float32)

        mapped_pos = delta_raw[:3][pos_order] * pos_sign * pos_scale
        mapped_rot = delta_raw[3:6][rot_order] * rot_sign * rot_scale
        delta_pose = np.concatenate([mapped_pos, mapped_rot]).astype(np.float32)

        current_pose = self._read_eef_pose()
        if self._teleop_initial_pose is None:
            self._teleop_initial_pose = current_pose.copy()

        target_pose = self._teleop_initial_pose + delta_pose
        pose_safe = self._safety_filter_pose(target_pose, current_pose)

        if gripper_cmd is not None:
            self._send_gripper(gripper_cmd)

        if self.config.enable_motion:
            self._send_eef_movep_canfd(pose_safe)

        self._last_pose = pose_safe.copy()

        if gripper_cmd is not None:
            if self.config.use_dummy_action:
                return np.concatenate(
                    [pose_safe, np.array([0.0], dtype=np.float32), np.array([gripper_cmd], dtype=np.float32)],
                    axis=0,
                )
            return np.concatenate([pose_safe, np.array([gripper_cmd], dtype=np.float32)], axis=0)
        return pose_safe

    def _read_joint_deg(self) -> np.ndarray:
        ret = self.arm.rm_get_joint_degree()
        if not isinstance(ret, tuple) or len(ret) < 2:
            raise RuntimeError(f"unexpected rm_get_joint_degree return: {ret}")
        code, joint = ret[0], ret[1]
        if code != 0:
            raise RuntimeError(f"rm_get_joint_degree failed: code={code}, joint={joint}")
        joint = np.asarray(joint, dtype=np.float32).reshape(-1)
        if joint.shape[0] < self.config.dof:
            raise RuntimeError(f"joint length < {self.config.dof}: {joint}")
        return joint[: self.config.dof]

    def _read_eef_pose(self) -> np.ndarray:
        ret = self.arm.rm_get_current_arm_state()
        if not isinstance(ret, tuple) or len(ret) < 2:
            raise RuntimeError(f"unexpected rm_get_current_arm_state return: {ret}")
        code, state = ret[0], ret[1]
        if code != 0:
            raise RuntimeError(f"rm_get_current_arm_state failed: code={code}, state={state}")
        if not isinstance(state, dict) or "pose" not in state:
            raise RuntimeError(f"state has no pose: {state}")
        pose = np.asarray(state["pose"], dtype=np.float32).reshape(-1)
        if pose.shape[0] < 6:
            raise RuntimeError(f"pose length < 6: {pose}")
        return pose[:6]

    def _send_joint_deg(self, joint6: np.ndarray):
        joints = [float(x) for x in np.asarray(joint6).reshape(-1)[:6]]
        return self.arm.rm_movej(joints, self.config.speed, 0, 0, self.config.block)

    def _send_eef_pose(self, pose6: np.ndarray):
        pose = [float(x) for x in np.asarray(pose6).reshape(-1)[:6]]
        try:
            return self.arm.rm_movej_p(pose, self.config.speed, 0, 0, self.config.block)
        except Exception:
            return self.arm.rm_movel(pose, self.config.speed, 0, 0, self.config.block)

    def _send_eef_movep_canfd(self, pose6: np.ndarray):
        pose = [float(x) for x in np.asarray(pose6).reshape(-1)[:6]]
        return self.arm.rm_movep_canfd(pose, follow=False)

    def _send_eef_velocity_towards(self, current_pose: np.ndarray, target_pose: np.ndarray):
        current = np.asarray(current_pose, dtype=np.float32).reshape(-1)[:6]
        target = np.asarray(target_pose, dtype=np.float32).reshape(-1)[:6]
        delta = target - current
        linear = delta[:3] * float(self.config.teleop_velocity_gain)
        angular = delta[3:6] * float(self.config.teleop_angular_velocity_gain)
        linear = np.clip(linear, -float(self.config.max_cartesian_velocity_mps), float(self.config.max_cartesian_velocity_mps))
        angular = np.clip(angular, -float(self.config.max_angular_velocity_rps), float(self.config.max_angular_velocity_rps))
        twist = np.concatenate([linear, angular]).astype(np.float32)
        return self.arm.rm_movev_canfd(twist.tolist(), False, 0, 0)

    def reset_to_initial(self):
        joint6 = [float(x) for x in self.config.initial_joint6]
        ret = self.arm.rm_movej(joint6, self.config.reset_speed, 0, 0, True)
        print(f"[RM65 RESET RET] {ret}")
        return ret

    def _read_gripper(self) -> float:
        if not self.config.use_gripper:
            return 1.0
        gripper = self.gripper.get()
        if gripper < 0:
            return 1.0
        return float(gripper)

    def _send_gripper(self, gripper_cmd: float) -> bool:
        cmd = float(np.clip(gripper_cmd, self.config.gripper_min, self.config.gripper_max))
        return self.gripper.send(cmd)

    def _clip_pose_workspace(self, pose: np.ndarray) -> np.ndarray:
        pose = pose.copy()
        pose[0] = float(np.clip(pose[0], self.config.x_min, self.config.x_max))
        pose[1] = float(np.clip(pose[1], self.config.y_min, self.config.y_max))
        pose[2] = float(np.clip(pose[2], self.config.z_min, self.config.z_max))
        return pose

    def _safety_filter_pose(self, raw_pose_cmd: np.ndarray, current_pose: np.ndarray | None) -> np.ndarray:
        pose = np.asarray(raw_pose_cmd, dtype=np.float32).reshape(-1)[:6]
        if pose.shape[0] != 6:
            raise ValueError(f"pose_cmd shape error: {pose.shape}")
        if not np.all(np.isfinite(pose)):
            raise ValueError(f"pose_cmd contains NaN/Inf: {pose}")
        pose = self._clip_pose_workspace(pose)
        if current_pose is not None:
            cur = np.asarray(current_pose, dtype=np.float32).reshape(-1)[:6]
            if cur.shape[0] == 6 and np.all(np.isfinite(cur)):
                max_delta_pos = float(self.config.max_delta_pos_m)
                max_delta_rot = float(self.config.max_delta_rot_rad)
                if max_delta_pos > 0 or max_delta_rot > 0:
                    delta = pose - cur
                    if max_delta_pos > 0:
                        delta[:3] = np.clip(delta[:3], -max_delta_pos, max_delta_pos)
                    if max_delta_rot > 0:
                        delta[3:6] = np.clip(delta[3:6], -max_delta_rot, max_delta_rot)
                    pose = cur + delta
                    pose = self._clip_pose_workspace(pose)
        return pose.astype(np.float32)

    def _clip_joint_action(self, target_joint_deg: np.ndarray) -> np.ndarray:
        target_joint_deg = np.asarray(target_joint_deg, dtype=np.float32).reshape(-1)
        if target_joint_deg.shape[0] != self.config.dof:
            raise ValueError(f"Expected joint action shape ({self.config.dof},), but got {target_joint_deg.shape}")
        max_step = float(self.config.max_relative_joint_step_deg)
        if max_step <= 0:
            return target_joint_deg.astype(np.float32)
        if self._last_joint_deg is None:
            current_joint_deg = self._read_joint_deg()
        else:
            current_joint_deg = self._last_joint_deg
        lower = current_joint_deg - max_step
        upper = current_joint_deg + max_step
        return np.clip(target_joint_deg, lower, upper).astype(np.float32)


Realman = RealmanRobot
