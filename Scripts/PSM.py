import omni.kit.commands
import omni.usd
import omni.isaac.core.utils.stage as stage_utils
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.world import World
from pxr import UsdPhysics, PhysxSchema, Gf, UsdGeom
import carb



class PSMRobot:
    def __init__(self, usd_path: str, pos: tuple = (0, 0, 0), psm_path: str = "/World/PSM", high_pd: bool = False):
        '''
            Initializes dVRK PSM

            Args:
                usd_path (str): path to psm_col.usd file in orbit surgical
                pos (tuple): Init position (x, y, z)
                high_pd (bool): whether to use high PD config
        '''

        self.usd_path = usd_path
        self.pos = pos
        self.high_pd = high_pd
        self.psm_path = psm_path

        self.stage = None
        self.world = None
        self.psm_prim = None
        self.articulation_controller = None

        self.init_joint_positions = {
            "psm_yaw_joint": 0.01,
            "psm_pitch_end_joint": 0.01,
            "psm_main_insertion_joint": 0.07,
            "psm_tool_roll_joint": 0.01,
            "psm_tool_pitch_joint": 0.01,
            "psm_tool_yaw_joint": 0.01,
            "psm_tool_gripper1_joint": -0.09,
            "psm_tool_gripper2_joint": 0.09
        }

        self.psm_actuator_config = {
            "joint_names": [
                "psm_yaw_joint",
                "psm_pitch_end_joint",
                "psm_main_insertion_joint",
                "psm_tool_roll_joint",
                "psm_tool_pitch_joint",
                "psm_tool_yaw_joint"
            ],
            "effort_limit": 12.0,
            "velocity_limit": 1.0,
            "stiffness": 800.0,
            "damping": 40.0
        }

        self.psm_tool_actuator_config = {
            "joint_names": [
                "psm_tool_gripper1_joint",
                "psm_tool_gripper2_joint"
            ],
            "effort_limit": 0.1,
            "velocity_limit": 0.2,
            "stiffness": 500.0,
            "damping": 0.1
        }


        if self.high_pd:
            self.psm_actuator_config["stiffness"] = 800.0
            self.psm_actuator_config["damping"] = 40.0
        
    def setup(self):
        self.stage = omni.usd.get_context().get_stage()
        if not self.stage:
            self.stage = stage_utils.create_new_stage()

        _ = omni.kit.commands.execute(
            "CreateReferenceCommand",
            path_to=self.psm_path,
            asset_path=self.usd_path,
            usd_context=omni.usd.get_context()
        )

        self.psm_prim = self.stage.GetPrimAtPath(self.psm_path)
        if not self.psm_prim:
            raise Exception(f"Failed to load PSM Prim at path: {self.psm_path}")
        
        if not self.psm_prim.HasAttribute("xformOp:translate"):
            UsdGeom.Xformable(self.psm_prim).AddTranslateOp()
        self.psm_prim.GetAttribute("xformOp:translate").Set(self.pos)

        self._configure_physics()
        self._configure_joints()

    def _configure_joints(self):
        for joint_name in self.psm_actuator_config["joint_names"]:
            joint_path = f"{self.psm_path}/{joint_name}"
            joint_prim = self.stage.GetPrimAtPath(joint_path)
            if joint_prim:
                drive_api = UsdPhysics.DriveAPI.Apply(joint_prim, "angular")
                drive_api.CreateTypeAttr().Set("force")
                drive_api.CreateDampingAttr().Set(self.psm_actuator_config["damping"])
                drive_api.CreateStiffnessAttr().Set(self.psm_actuator_config["stiffness"])
                drive_api.CreateMaxForceAttr().Set(self.psm_actuator_config["effort_limit"])

        for joint_name in self.psm_tool_actuator_config["joint_names"]:
            joint_path = f"{self.psm_path}/{joint_name}"
            joint_prim = self.stage.GetPrimAtPath(joint_path)
            if joint_prim:
                drive_api = UsdPhysics.DriveAPI.Apply(joint_prim, "angular")
                drive_api.CreateTypeAttr().Set("force")
                drive_api.CreateDampingAttr().Set(self.psm_actuator_config["damping"])
                drive_api.CreateStiffnessAttr().Set(self.psm_actuator_config["stiffness"])
                drive_api.CreateMaxForceAttr().Set(self.psm_actuator_config["effort_limit"])


    def _configure_physics(self):
        physx_articulation_api = PhysxSchema.PhysxArticulationAPI.Apply(self.psm_prim)
        physx_articulation_api.CreateEnabledSelfCollisionsAttr().Set(False)
        physx_articulation_api.CreateSolverPositionIterationCountAttr().Set(4)
        physx_articulation_api.CreateSolverVelocityIterationCountAttr().Set(0)

        scene_path = "/World/PhysicsScene"
        scene_prim = self.stage.GetPrimAtPath(scene_path)
        if not scene_prim:
            scene_prim = self.stage.DefinePrim(scene_path)
            physics_scene = UsdPhysics.Scene.Defien(self.stage, scene_path)
        else:
            physics_scene = UsdPhysics.Scene.Get(self.stage, scene_path)

        if physics_scene:
            physics_scene.CreateGravityDirectionAttr().Set((0, 0, -1))

            if self.high_pd:
                physics_scene.CreateGravityMagnitudeAttr().Set(0.0)
            else:
                physics_scene.CreateGravityMagnitudeAttr().Set(9.81)

        UsdPhysics.Scene.Define(self.stage, scene_path)

    def setup_articulation_controller(self):
        self.world = World.instance()
        if not self.world:
            self.world = World()

        self.articulation_view = ArticulationView(prim_paths_expr=self.psm_prim, name="psm_view")
        self.world.scene.add(self.articulation_view)

        self.articulation_controller = self.articulation_view.get_articul