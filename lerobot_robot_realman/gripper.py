import numpy as np


try:
    from lerobot.robots.my_robot.controller.HitbotPicker_controller import HitbotPickerController
except ImportError:
    HitbotPickerController = None


class HitbotGripper:
    def __init__(self, port: str = "/dev/ttyUSB0", enable: bool = False, deadband: float = 0.02):
        self.port = port
        self.enable = enable
        self.deadband = deadband
        self.gripper = None
        self.initialized = False
        self.last_cmd = None
        self.verbose = False

    def connect(self) -> bool:
        if not self.enable:
            print("[GRIPPER SAFE MODE] enable_gripper=False，不初始化外接夹爪。")
            return False
        if HitbotPickerController is None:
            raise ImportError("Cannot import HitbotPickerController.")
        try:
            self.gripper = HitbotPickerController(port=self.port)
            self.initialized = bool(self.gripper.set_up())
            if self.initialized:
                print(f"[HitbotGripper] initialized on {self.port}")
            return self.initialized
        except Exception as e:
            print(f"[HitbotGripper] connect exception: {e}")
            self.initialized = False
            return False

    def get(self) -> float:
        if not self.initialized or self.gripper is None:
            return -1.0
        try:
            joint_state = self.gripper.get_joint()
            if joint_state is not None and len(joint_state) > 0:
                return float(joint_state[0])
        except Exception as e:
            print(f"[HitbotGripper] get failed: {e}")
        return -1.0

    def send(self, gripper_cmd: float) -> bool:
        cmd = float(np.clip(gripper_cmd, 0.0, 1.0))
        if not self.enable or not self.initialized or self.gripper is None:
            return False
        if self.last_cmd is not None and abs(cmd - self.last_cmd) < self.deadband:
            return True
        try:
            ret = self.gripper.set_joint(cmd)
            self.last_cmd = cmd
            return bool(ret)
        except Exception as e:
            print(f"[HitbotGripper] send failed: {e}")
            return False

    def close_device(self) -> None:
        try:
            if self.initialized and self.gripper is not None:
                self.gripper.set_joint(1.0)
        except Exception:
            pass
