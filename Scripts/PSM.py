import omni.kit.commands
import omni.usd
import omni.isaac.core.utils.stage as stage_utils
import omni.isaac.core.utils.prims as prim_utils
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.world import World
from pxr import UsdPhysics, PhysxSchema, Gf
import carb

class PSMRobot:
    def __init__(self, usd_path, position=(0.0, 0.0, 0.15), high_pd=False):
        """Initialize the dVRK PSM Robot.
        
        Args:
            usd_path (str): Path to the PSM USD file
            position (tuple): Initial position (x, y, z) of the robot
            high_pd (bool): Whether to use high PD configuration
        """
        self.usd_path = usd_path
        self.position = position
        self.high_pd = high_pd
        self.stage = None
        self.psm_prim = None
        self.articulation_controller = None
        self.world = None
        
        # Initial joint positions
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
        
        # Actuator configuration
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
        
        # Apply high PD configuration
        if self.high_pd:
            self.psm_actuator_config["stiffness"] = 800.0
            self.psm_actuator_config["damping"] = 40.0
        
    def setup(self):
        """Set up the stage and load the robot"""
        self.stage = omni.usd.get_context().get_stage()
        if not self.stage:
            self.stage = stage_utils.create_new_stage()
        
        psm_path = "/World/PSM"
        result = omni.kit.commands.execute(
            "CreateReferenceCommand",
            path_to=psm_path,
            asset_path=self.usd_path,
            usd_context=omni.usd.get_context()
        )
        
        self.psm_prim = self.stage.GetPrimAtPath(psm_path)
        if not self.psm_prim:
            raise Exception(f"Failed to load PSM robot at path: {psm_path}")
        
        # Fix for translation_local error - use USD API instead of command
        xform = self.psm_prim.GetAttribute("xformOp:translate")
        if xform:
            xform.Set(Gf.Vec3d(self.position))
        else:
            # Add transform if it doesn't exist
            from pxr import UsdGeom
            xformable = UsdGeom.Xformable(self.psm_prim)
            xformable.AddTranslateOp().Set(Gf.Vec3d(self.position))
        
        self._configure_physics()
        self._configure_joints()
        
        return self.psm_prim
        
    def _configure_physics(self):
        """Configure physics properties for the robot"""
        root_path = self.psm_prim.GetPath()
        
        # Add PhysX articulation API
        physx_articulation_api = PhysxSchema.PhysxArticulationAPI.Apply(self.psm_prim)
        physx_articulation_api.CreateEnabledSelfCollisionsAttr().Set(False)
        physx_articulation_api.CreateSolverPositionIterationCountAttr().Set(4)
        physx_articulation_api.CreateSolverVelocityIterationCountAttr().Set(0)
        
        # Ensure physics scene is properly created and set as active
        stage = self.psm_prim.GetStage()
        scene_path = "/World/PhysicsScene"
        scene_prim = stage.GetPrimAtPath(scene_path)
        if not scene_prim:
            scene_prim = stage.DefinePrim(scene_path, "Xform")
            physics_scene = UsdPhysics.Scene.Define(stage, scene_path)
        else:
            physics_scene = UsdPhysics.Scene.Get(stage, scene_path)
        
        # Set gravity (disable gravity for high PD config)
        if physics_scene:
            physics_scene.CreateGravityDirectionAttr().Set((0, 0, -1))
            if self.high_pd:
                physics_scene.CreateGravityMagnitudeAttr().Set(0.0)
            else:
                physics_scene.CreateGravityMagnitudeAttr().Set(9.81)
        
        # Make sure physics scene is active
        UsdPhysics.Scene.Define(stage, scene_path)
    
    def _configure_joints(self):
        """Configure joint drives for the robot"""
        for joint_name in self.psm_actuator_config["joint_names"]:
            joint_path = f"{self.psm_prim.GetPath()}/{joint_name}"
            joint_prim = self.stage.GetPrimAtPath(joint_path)
            
            if joint_prim:
                drive_api = UsdPhysics.DriveAPI.Apply(joint_prim, "angular")
                drive_api.CreateTypeAttr().Set("force")
                drive_api.CreateDampingAttr().Set(self.psm_actuator_config["damping"])
                drive_api.CreateStiffnessAttr().Set(self.psm_actuator_config["stiffness"])
                drive_api.CreateMaxForceAttr().Set(self.psm_actuator_config["effort_limit"])
                
                if joint_name in self.init_joint_positions:
                    # This will be set at runtime through the articulation controller
                    pass
        
        for joint_name in self.psm_tool_actuator_config["joint_names"]:
            joint_path = f"{self.psm_prim.GetPath()}/{joint_name}"
            joint_prim = self.stage.GetPrimAtPath(joint_path)
            
            if joint_prim:
                drive_api = UsdPhysics.DriveAPI.Apply(joint_prim, "angular")
                drive_api.CreateTypeAttr().Set("force")
                drive_api.CreateDampingAttr().Set(self.psm_tool_actuator_config["damping"])
                drive_api.CreateStiffnessAttr().Set(self.psm_tool_actuator_config["stiffness"])
                drive_api.CreateMaxForceAttr().Set(self.psm_tool_actuator_config["effort_limit"])
    
    def setup_articulation_controller(self):
        """Set up the articulation controller for controlling the robot at runtime"""
        # Fix for articulation controller error - initialize World properly
        if World.instance() is None:
            self.world = World()
        else:
            self.world = World.instance()
            
        
        # Fix - Make sure prim path is a string
        prim_path_str = str(self.psm_prim.GetPath())
        
        # Create the articulation view
        self.articulation_view = ArticulationView(
            prim_paths_expr=prim_path_str, 
            name="psm_view"
        )
        
        self.world.scene.add(self.articulation_view)
        
        # Make sure the articulation view is properly initialized
        self.world.reset()
        
        # Get the articulation controller
        self.articulation_controller = self.articulation_view.get_articulation_controller()
        
        joint_indices = {}
        joint_positions = []
        
        all_joint_names = (
            self.psm_actuator_config["joint_names"] + 
            self.psm_tool_actuator_config["joint_names"]
        )
        
        for joint_name in all_joint_names:
            if joint_name in self.init_joint_positions:
                try:
                    joint_indices[joint_name] = self.articulation_view.get_dof_index(joint_name)
                    joint_positions.append(self.init_joint_positions[joint_name])
                except Exception as e:
                    carb.log_warn(f"Could not get DOF index for joint {joint_name}: {e}")
        
        if joint_positions and joint_indices:
            try:
                self.articulation_controller.apply_joint_positions(
                    positions=joint_positions, 
                    joint_indices=list(joint_indices.values())
                )
            except Exception as e:
                carb.log_warn(f"Could not apply joint positions: {e}")
        
        return self.articulation_controller
    
    def setup_ros2(self):
        """Set up ROS 2 bridge for the robot"""
        # First check if the extension is available
        try:
            from omni.isaac.core.utils.extensions import enable_extension, get_extension_path_from_name
            
            # Check if ROS2 extension exists before enabling
            if get_extension_path_from_name("omni.isaac.ros2_bridge"):
                enable_extension("omni.isaac.ros2_bridge")
                carb.log_info("ROS2 bridge extension enabled")
            else:
                carb.log_warn("omni.isaac.ros2_bridge extension not found. ROS2 functionality will not be available.")
                return False
            
            # Try to import ROS2 modules using the updated Isaac Sim 4.2 paths
            try:
                # Import Isaac ROS 2 bridge - try different import paths
                try:
                    import omni.isaac.ros2_bridge
                    carb.log_info("Successfully imported omni.isaac.ros2_bridge")
                    
                    # Try to import the clock publisher
                    try:
                        from omni.isaac.ros2_bridge.isaac_ros_tools import IsaacROSClockPublisher
                        self.clock_publisher = IsaacROSClockPublisher()
                        self.clock_publisher.initialize()
                        carb.log_info("ROS2 clock publisher set up successfully")
                    except ImportError:
                        carb.log_warn("Could not import IsaacROSClockPublisher. Clock will not be published.")
                    
                    # Try to import joint state publisher
                    try:
                        from omni.isaac.ros2_bridge.isaac_ros_tools import IsaacROSJointStatePublisher
                        self.joint_state_publisher = IsaacROSJointStatePublisher(
                            prim_path=str(self.psm_prim.GetPath()),
                            topic_name="/joint_states",
                            publishing_rate=30
                        )
                        self.joint_state_publisher.initialize()
                        carb.log_info("Joint state publisher set up successfully")
                    except ImportError:
                        carb.log_warn("Could not import IsaacROSJointStatePublisher. Joint states will not be published.")
                    
                    carb.log_info("ROS 2 bridge setup complete for PSM robot")
                    return True
                    
                except ImportError as e:
                    carb.log_warn(f"Could not import omni.isaac.ros2_bridge: {e}")
                    
                    # Try the older version (legacy support)
                    try:
                        import omni.isaac.ros_bridge
                        carb.log_info("Found legacy omni.isaac.ros_bridge instead")
                        
                        # For legacy support
                        from omni.isaac.ros_bridge.clock import ROSClock
                        self.clock_publisher = ROSClock()
                        self.clock_publisher.setup()
                        
                        carb.log_info("Legacy ROS bridge setup complete for PSM robot")
                        return True
                    except ImportError:
                        carb.log_warn("Legacy ROS bridge not found either")
                        return False
                    
            except Exception as e:
                carb.log_warn(f"Error during ROS 2 bridge module import: {e}")
                return False
                
        except Exception as e:
            carb.log_warn(f"Error setting up ROS 2 bridge: {e}")
            carb.log_warn("ROS2 functionality will not be available")
            return False

def spawn_psm_robot(usd_path, position=(0.0, 0.0, 0.15), high_pd=False, setup_ros2=True):
    """Spawn a PSM robot with the given configuration
    
    Args:
        usd_path (str): Path to the PSM USD file
        position (tuple): Initial position (x, y, z) of the robot
        high_pd (bool): Whether to use high PD configuration
        setup_ros2 (bool): Whether to set up ROS 2 bridge
    
    Returns:
        PSMRobot: The PSM robot instance
    """
    psm_robot = PSMRobot(usd_path=usd_path, position=position, high_pd=high_pd)
    
    try:
        psm_prim = psm_robot.setup()
        
        config_type = "High PD" if high_pd else "Standard"
        carb.log_info(f"{config_type} PSM robot loaded successfully at position {position}")
    except Exception as e:
        carb.log_error(f"Failed to set up PSM robot: {e}")
        return None
    
    try:
        controller = psm_robot.setup_articulation_controller()
        carb.log_info("Articulation controller set up successfully")
    except Exception as e:
        carb.log_error(f"Failed to set up articulation controller: {e}")
        # Continue despite error
    
    # Set up ROS 2 bridge if requested
    if setup_ros2:
        try:
            ros2_success = psm_robot.setup_ros2()
            status = "successfully" if ros2_success else "with some limitations"
            carb.log_info(f"ROS 2 bridge set up {status}")
        except Exception as e:
            carb.log_error(f"Failed to set up ROS 2 bridge: {e}")
            # Continue despite error
    
    return psm_robot


# Usage example:
if __name__ == "__main__":
    psm_col_usd = r"/home/imad/orbit-surgical/source/extensions/orbit.surgical.assets/data/Robots/dVRK/PSM/psm_col.usd"
    
    # Load the robot with high PD configuration
    robot = spawn_psm_robot(psm_col_usd, high_pd=True)