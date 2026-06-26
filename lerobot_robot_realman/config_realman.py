from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig
from lerobot.cameras.configs import ColorMode, Cv2Rotation
from lerobot.cameras.realsense import RealSenseCameraConfig
from lerobot.robots.config import RobotConfig


@RobotConfig.register_subclass("lerobot_robot_realman")
@dataclass
class RealmanConfig(RobotConfig):
    ip: str = "192.168.1.18"
    port: int = 8080
    connection_level: int = 3
    dof: int = 6

    control_mode: str = "delta_eef"
    use_dummy_action: bool = False
    delta_eef_position_scale: tuple[float, float, float] = (4.0, 4.0, 4.0)
    delta_eef_rotation_scale: tuple[float, float, float] = (1.5, 1.5, 1.5)
    delta_eef_position_axis_order: tuple[int, int, int] = (0, 1, 2)
    delta_eef_position_axis_sign: tuple[float, float, float] = (1.0, 1.0, 1.0)
    delta_eef_rotation_axis_order: tuple[int, int, int] = (1, 0, 2)
    delta_eef_rotation_axis_sign: tuple[float, float, float] = (-1.0, 1.0, 1.0)

    enable_motion: bool = True
    speed: int = 10
    reset_speed: int = 10
    block: bool = False
    reset_on_connect: bool = True
    sdk_timeout_ms: int = 1000
    enable_movev_canfd_init: bool = True
    movev_canfd_follow: int = 1
    movev_canfd_frame_type: int = 1
    movev_canfd_dt_ms: int = 10
    teleop_velocity_gain: float = 6.0
    teleop_angular_velocity_gain: float = 2.0
    max_cartesian_velocity_mps: float = 0.25
    max_angular_velocity_rps: float = 0.8
    enable_camera_observation: bool = False
    verbose_motion_logs: bool = False

    reset_on_disconnect: bool = False
    initial_joint6: tuple[float, float, float, float, float, float] = (
        0.0, 0.0, 90.0, 0.0, 90.0, 0.0
    )

    use_gripper: bool = True
    enable_gripper: bool = True
    gripper_port: str = "/dev/ttyUSB0"
    gripper_min: float = 0.0
    gripper_max: float = 1.0
    gripper_cmd_deadband: float = 0.0

    x_min: float = -0.45
    x_max: float = 0.10
    y_min: float = -0.35
    y_max: float = 0.35
    z_min: float = 0.12
    z_max: float = 0.60
    max_delta_pos_m: float = 0.0
    max_delta_rot_rad: float = 0.0
    max_relative_joint_step_deg: float = 0.0

    cameras: dict[str, CameraConfig] = field(
        default_factory=lambda: {
            "cam_head": RealSenseCameraConfig(
                serial_number_or_name="135522078943",
                fps=30,
                width=640,
                height=480,
                color_mode=ColorMode.RGB,
                rotation=Cv2Rotation.NO_ROTATION,
            ),
        }
    )
