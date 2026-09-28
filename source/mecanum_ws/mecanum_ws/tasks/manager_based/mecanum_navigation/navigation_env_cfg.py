# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""End-to-end point-goal navigation of a mecanum robot among pillars (cylindrical obstacles)."""

import math

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from mecanum_ws.robots import mecanum

from . import mdp

##
# Task parameters
##

TILE_SIZE = 10.0
"""Side length of one terrain tile [m]. Robots spawn in its center, goals are sampled inside it."""

LIDAR_MAX_DISTANCE = 5.0
"""Range of the planar lidar [m]."""

LIDAR_SENSOR = SceneEntityCfg("lidar")

WHEELS = SceneEntityCfg("robot", joint_names=mecanum.WHEEL_JOINT_NAMES, preserve_order=True)

EPISODE_LENGTH_S = 12.0
"""Episode length = time the robot has to reach the goal [s] (goals are up to ~6.4 m away)."""

TASK_REWARD_DURATION = 2.0
"""Duration ``T_r`` of the final-position task reward at the end of the episode [s]."""

##
# Scene definition
##

PILLARS_TERRAIN_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(TILE_SIZE, TILE_SIZE),
    border_width=5.0,
    num_rows=10,  # difficulty levels (curriculum): more and thicker pillars on higher rows
    num_cols=20,  # random variants per level
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=True,
    sub_terrains={
        "pillars": terrain_gen.MeshRepeatedCylindersTerrainCfg(
            proportion=1.0,
            # pillar-free square in the tile center where the robots spawn
            platform_width=3.0,
            # a (almost) flat platform; a negative value would raise it to the pillar height
            platform_height=0.01,
            object_params_start=terrain_gen.MeshRepeatedCylindersTerrainCfg.ObjectCfg(
                num_objects=4, height=1.5, radius=0.15
            ),
            object_params_end=terrain_gen.MeshRepeatedCylindersTerrainCfg.ObjectCfg(
                num_objects=30, height=1.5, radius=0.30
            ),
            # goal candidates: points that are at least `patch_radius` away from any pillar
            flat_patch_sampling={
                "target": terrain_gen.FlatPatchSamplingCfg(
                    num_patches=64,
                    patch_radius=0.7,
                    max_height_diff=0.05,
                    x_range=(-0.45 * TILE_SIZE, 0.45 * TILE_SIZE),
                    y_range=(-0.45 * TILE_SIZE, 0.45 * TILE_SIZE),
                )
            },
        ),
    },
)
"""Tiles of flat ground with randomly placed vertical cylinders (pillars)."""


@configclass
class MecanumPillarsSceneCfg(InteractiveSceneCfg):
    """Ground with pillars, a mecanum robot and a planar 360° lidar."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=PILLARS_TERRAIN_CFG,
        max_init_terrain_level=2,
        collision_group=-1,
        # carries the compliant roller contact (only effective on the ground side)
        physics_material=mecanum.ground_material(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.45, 0.45, 0.5)),
        debug_vis=False,
    )

    robot: ArticulationCfg = mecanum.MECANUM_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # planar lidar: 72 horizontal rays (every 5°), cast only against the terrain mesh (ground + pillars)
    lidar = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/" + mecanum.BASE_LINK_NAME,
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.15)),
        ray_alignment="yaw",
        pattern_cfg=patterns.LidarPatternCfg(
            channels=1, vertical_fov_range=(0.0, 0.0), horizontal_fov_range=(-180.0, 180.0), horizontal_res=5.0
        ),
        max_distance=LIDAR_MAX_DISTANCE,
        mesh_prim_paths=["/World/ground"],
        debug_vis=False,
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=750.0),
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command specifications for the MDP: a goal position on the tile, fixed for the whole episode."""

    # debug vis: red sphere at the goal, green arrow above each robot pointing at its goal
    goal_pose = mdp.GoalPositionCommandCfg(
        asset_name="robot",
        simple_heading=True,  # heading is not used by the task (holonomic robot)
        resampling_time_range=(1.0e9, 1.0e9),  # only resampled on reset
        debug_vis=True,
        ranges=mdp.GoalPositionCommandCfg.Ranges(heading=(-math.pi, math.pi)),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP: the policy commands the four wheel speed setpoints.

    The setpoints go through the torque-limited DC motor model of the wheels, so the policy faces the roller physics
    directly: the wheels slip whenever the torque demand exceeds the traction.
    """

    wheel_vel = mdp.JointVelocityActionCfg(
        asset_name="robot",
        joint_names=mecanum.WHEEL_JOINT_NAMES,
        scale=mecanum.MAX_WHEEL_SPEED,
        use_default_offset=False,
        preserve_order=True,
        clip={".*": (-mecanum.MAX_WHEEL_SPEED, mecanum.MAX_WHEEL_SPEED)},
    )


@configclass
class IkActionsCfg:
    """Action specifications for the MDP: the policy commands a base twist, mapped to wheels by mecanum IK."""

    base_vel = mdp.MecanumBaseVelocityActionCfg(
        asset_name="robot",
        wheel_joint_names=mecanum.WHEEL_JOINT_NAMES,
        wheel_joint_signs=mecanum.WHEEL_JOINT_SIGNS,
        wheel_radius=mecanum.WHEEL_RADIUS,
        wheel_base_half_length=mecanum.WHEEL_BASE_HALF_LENGTH,
        track_half_width=mecanum.TRACK_HALF_WIDTH,
        scale=(1.0, 1.0, 1.5),
        max_wheel_speed=mecanum.MAX_WHEEL_SPEED,
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.05, n_max=0.05))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        goal_position = ObsTerm(
            func=mdp.goal_position_b, params={"command_name": "goal_pose"}, noise=Unoise(n_min=-0.05, n_max=0.05)
        )
        time_left = ObsTerm(func=mdp.time_left)
        lidar = ObsTerm(
            func=mdp.lidar_distances,
            params={"sensor_cfg": LIDAR_SENSOR, "max_distance": LIDAR_MAX_DISTANCE},
            scale=1.0 / LIDAR_MAX_DISTANCE,
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        wheel_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=mecanum.WHEEL_JOINT_NAMES, preserve_order=True)},
            scale=1.0 / mecanum.MAX_WHEEL_SPEED,
            noise=Unoise(n_min=-0.02, n_max=0.02),
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            # wide range: the real roller-floor friction is unknown
            "static_friction_range": (0.5, 1.1),
            "dynamic_friction_range": (0.4, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    # reset: random pose on the pillar-free center platform of the tile
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.3, 0.3), "y": (-0.3, 0.3), "yaw": (-math.pi, math.pi)},
            "velocity_range": {},
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP, following Rudin et al. 2022 ("Advanced Skills by Learning Locomotion and Local
    Navigation End-to-End", arXiv:2209.12827).

    The only task reward is given in the last ``TASK_REWARD_DURATION`` seconds of the episode and depends on the final
    distance to the goal, so the policy is free to choose its path and speed. The penalties do not compete with the
    task as long as the goal is reached in time. Note: the reward manager multiplies every term by the step time
    (7/360 s), so a term that is 1 for the whole episode sums up to ``EPISODE_LENGTH_S * weight``.
    """

    # task, eq. (1): episode sum is at most `weight` (robot at the goal during the whole reward window)
    final_position = RewTerm(
        func=mdp.final_position,
        weight=10.0,
        params={"command_name": "goal_pose", "reward_duration": TASK_REWARD_DURATION, "std": 1.0},
    )
    # exploration, eq. (3): removed automatically once the task reward reaches 50 % of its maximum
    exploration_bias = RewTerm(
        func=mdp.exploration_bias,
        weight=0.5,
        params={"command_name": "goal_pose", "reward_duration": TASK_REWARD_DURATION, "threshold": 0.5, "std": 1.0},
    )
    # ---- penalties: all off (weight 0 -> skipped by the reward manager). scripts/tools/staged_penalties.py switches
    # them on one at a time via Hydra (env.rewards.<term>.weight=...). Hitting a pillar or flipping over terminates the
    # episode (TerminationsCfg), which forfeits the task reward. "was": weight used before the penalties were removed.
    # -- joint torques (Rudin eq. 2) -> wheel motor torques; motor heat E_e ~ tau^2 (Xie et al. 2020); was -2e-4
    wheel_torque_l2 = RewTerm(func=mdp.joint_torques_l2, weight=0.0, params={"asset_cfg": WHEELS})
    # -- abrupt action changes (Rudin eq. 2; Finke et al. 2026); was -0.01
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=0.0)
    # -- joint accelerations (Rudin eq. 2) -> wheel accelerations; kinetic energy changes E_k (Xie); was -1e-6
    wheel_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=0.0, params={"asset_cfg": WHEELS})
    # -- stalling (Rudin eq. 4): < 0.1 m/s farther than 0.5 m from the goal; ~ idle energy E_idle (Xie); was -0.5
    stalling = RewTerm(func=mdp.stalling, weight=0.0, params={"command_name": "goal_pose"})
    # -- wheel slip on the rollers; friction dissipation E_f (Xie); stands in for the feet accelerations; was -0.5
    wheel_slip_l2 = RewTerm(
        func=mdp.wheel_slip_l2,
        weight=0.0,
        params={
            "asset_cfg": WHEELS,
            "wheel_radius": mecanum.WHEEL_RADIUS,
            "wheel_base_half_length": mecanum.WHEEL_BASE_HALF_LENGTH,
            "track_half_width": mecanum.TRACK_HALF_WIDTH,
            "wheel_joint_signs": mecanum.WHEEL_JOINT_SIGNS,
        },
    )
    # -- action magnitude (Finke et al. 2026: -0.05)
    action_l2 = RewTerm(func=mdp.action_l2, weight=0.0)
    # -- obstacles (pillar terrain): proximity (not in the papers; was -1.0) and collision (Rudin eq. 2; was -100)
    obstacle_proximity = RewTerm(
        func=mdp.obstacle_proximity,
        weight=0.0,
        params={
            "sensor_cfg": LIDAR_SENSOR,
            "max_distance": LIDAR_MAX_DISTANCE,
            "half_extents": mecanum.FOOTPRINT_HALF_EXTENTS,
            "safe_clearance": 0.25,
        },
    )
    collision = RewTerm(func=mdp.is_terminated_term, weight=0.0, params={"term_keys": ["collision", "flipped"]})
    # -- chassis roll / pitch rates (not in the papers); was -0.05
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=0.0)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # not a truncation: the remaining time is observed, so (as in the paper) the value is not bootstrapped at the end
    time_out = DoneTerm(func=mdp.time_out, time_out=False)
    collision = DoneTerm(
        func=mdp.obstacle_collision,
        params={
            "sensor_cfg": LIDAR_SENSOR,
            "max_distance": LIDAR_MAX_DISTANCE,
            "half_extents": mecanum.FOOTPRINT_HALF_EXTENTS,
            "margin": 0.03,
        },
    )
    flipped = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": math.radians(60.0)})


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    terrain_levels = CurrTerm(
        func=mdp.terrain_levels_goal, params={"command_name": "goal_pose", "success_radius": 0.5, "fail_radius": 2.0}
    )


##
# Viewer / video camera
##


def overview_camera(cfg: ManagerBasedRLEnvCfg, pitch_deg: float = 45.0, margin: float = 1.1) -> None:
    """Point the viewer (and the recorded videos) at the whole curriculum.

    The difficulty levels (terrain rows) run along x and the terrain is centered on the origin. The camera stands in
    front of the first terrain column (-y side) and looks along +y, far enough away that all levels fit into the
    60 deg horizontal field of view of the default camera: level 0 is on the left, the hardest level on the right.
    The goal markers are enlarged so that they stay visible from ~100 m.
    Note: it is computed from the terrain size at config time; Hydra overrides of the terrain size need matching
    ``env.viewer.eye`` / ``env.viewer.lookat`` overrides.
    """
    gen = cfg.scene.terrain.terrain_generator
    half_width = 0.5 * gen.num_rows * gen.size[0]
    distance = margin * half_width / math.tan(math.radians(30.0))
    lookat = (0.0, -0.5 * gen.num_cols * gen.size[1] + 1.5 * gen.size[1], 0.0)
    pitch = math.radians(pitch_deg)
    cfg.viewer.origin_type = "world"
    cfg.viewer.lookat = lookat
    cfg.viewer.eye = (0.0, lookat[1] - distance * math.cos(pitch), distance * math.sin(pitch))
    cfg.viewer.resolution = (1920, 1080)
    cmd = cfg.commands.goal_pose
    cmd.highlight_env_index = -1
    cmd.goal_visualizer_cfg.markers["goal"].radius = 0.35
    cmd.direction_visualizer_cfg.markers["arrow"].scale = (0.7, 0.7, 1.0)


def follow_camera(cfg: ManagerBasedRLEnvCfg, env_index: int = 1) -> None:
    """Follow the robot of ``env_index`` from above-behind; its goal and arrow are highlighted (yellow)."""
    cfg.viewer.eye = (-4.0, -4.0, 5.0)
    cfg.viewer.lookat = (0.0, 0.0, 0.0)
    cfg.viewer.origin_type = "asset_root"
    cfg.viewer.asset_name = "robot"
    cfg.viewer.env_index = env_index
    cfg.commands.goal_pose.highlight_env_index = env_index


##
# Environment configuration
##


@configclass
class MecanumNavigationPillarsEnvCfg(ManagerBasedRLEnvCfg):
    """Drive to a goal position among pillars; the policy outputs the four wheel speed setpoints."""

    scene: MecanumPillarsSceneCfg = MecanumPillarsSceneCfg(num_envs=4096, env_spacing=TILE_SIZE)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings: 360 Hz physics (validated: below it the roller contact noise is numerical), 51 Hz policy
        self.decimation = 7
        self.episode_length_s = EPISODE_LENGTH_S
        # simulation settings
        self.sim.dt = 1.0 / 360.0
        self.sim.render_interval = self.decimation
        # default material = material of the robot colliders (the URDF spawner binds none)
        self.sim.physics_material = mecanum.ROBOT_MATERIAL
        # the default PhysX GPU buffers are enough for the 336 sphere colliders per robot (checked at 1536 envs, no
        # overflow warning); raise gpu_max_rigid_patch_count etc. only if PhysX reports an overflow (costs GPU memory)
        # sensor update rate = policy rate
        self.scene.lidar.update_period = self.decimation * self.sim.dt
        # viewer / video camera: follows the robot of env 1 (overview_camera(self): all curriculum levels in view)
        follow_camera(self)


@configclass
class MecanumNavigationPillarsEnvCfg_PLAY(MecanumNavigationPillarsEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 32
        # a smaller terrain, spread over all difficulty levels
        self.scene.terrain.max_init_terrain_level = None
        self.scene.terrain.terrain_generator.num_rows = 5
        self.scene.terrain.terrain_generator.num_cols = 5
        self.scene.terrain.terrain_generator.curriculum = False
        self.scene.lidar.debug_vis = True
        self.curriculum.terrain_levels = None
        self.observations.policy.enable_corruption = False
        self.events.physics_material = None


@configclass
class MecanumNavigationPillarsIkEnvCfg(MecanumNavigationPillarsEnvCfg):
    """Same task; the policy outputs a base twist (vx, vy, wz) that is mapped to wheel setpoints by mecanum IK."""

    actions: IkActionsCfg = IkActionsCfg()


@configclass
class MecanumNavigationPillarsIkEnvCfg_PLAY(MecanumNavigationPillarsEnvCfg_PLAY):
    actions: IkActionsCfg = IkActionsCfg()


@configclass
class MecanumNavigationFlatEnvCfg(MecanumNavigationPillarsEnvCfg):
    """Same navigation task on flat ground: the terrain tiles have no pillars (the lidar sees nothing), so there is no
    curriculum either. A first step before learning to avoid obstacles."""

    def __post_init__(self) -> None:
        super().__post_init__()
        pillars = self.scene.terrain.terrain_generator.sub_terrains["pillars"]
        pillars.object_params_start.num_objects = 0
        pillars.object_params_end.num_objects = 0
        # all tiles are identical: spread the robots over all of them and keep them there
        self.scene.terrain.terrain_generator.curriculum = False
        self.scene.terrain.max_init_terrain_level = None
        self.curriculum.terrain_levels = None


@configclass
class MecanumNavigationFlatEnvCfg_PLAY(MecanumNavigationFlatEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 5
        self.scene.terrain.terrain_generator.num_cols = 5
        self.observations.policy.enable_corruption = False
        self.events.physics_material = None
