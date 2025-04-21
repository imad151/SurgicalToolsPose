import argparse
import os
import numpy as np
from PIL import Image
from omni.isaac.lab.app import AppLauncher

parser = argparse.ArgumentParser(description="Surgical tool capture with fixed robot position")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import omni.isaac.core.utils.prims as prim_utils
import omni.replicator.core as rep

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.sim import SimulationContext
from omni.isaac.lab.sensors import TiledCameraCfg, TiledCamera
from omni.isaac.lab.utils import convert_dict_to_backend

from orbit.surgical.assets.psm import PSM_HIGH_PD_CFG

from scipy.spatial.transform import Rotation as R


def design_scene() -> dict:
    print("[DEBUG] Starting scene design...")
    cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Lights", cfg)

    origins = [[0.0, 0.0, 0.0]]
    prim_utils.create_prim("/World/Origin", "Xform", translation=origins[0])
    print("[DEBUG] Created Origin prim")

    psm_cfg = PSM_HIGH_PD_CFG.copy()
    psm_cfg.prim_path = "/World/Origin/Robot"
    psm = Articulation(psm_cfg)
    print("[DEBUG] Created robot articulation at path:", psm_cfg.prim_path)

    # Modified camera position and orientation to view the tool tip from further away
    cam: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/Camera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0, 1, 0),  # Further away position
            rot=(0, -0.70711, -0.70711, -0),  # x z y w  
            convention='world'
        ),
        data_types=["rgb"],  # Only request rgb
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, 
            focus_distance=400.0, 
            horizontal_aperture=20.955, 
            clipping_range=(0.1, 1e5),
        ),
        height=480,
        width=640
    )

    cam = TiledCamera(cam)
    print("[DEBUG] Created camera at path:", cam.cfg.prim_path)
    print("[DEBUG] Camera data types:", cam.cfg.data_types)
    
    scene_entities = {"psm": psm,
                      "cam": cam}
    return scene_entities, origins

def run_simulator(sim: sim_utils.SimulationContext, entities: dict[str, Articulation], origins: torch.Tensor):
    robot = entities["psm"]
    cam = entities["cam"]
    sim_dt = sim.get_physics_dt()
    count = 0

    output_dir = r"/home/imad/SurgicalToolsPose/TestImages/"
    
    # Check if output directory exists and create it if it doesn't
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            print(f"[DEBUG] Created output directory: {output_dir}")
        except Exception as e:
            print(f"[ERROR] Failed to create output directory: {e}")
    else:
        print(f"[DEBUG] Output directory exists: {output_dir}")
    
    # Check write permissions on output directory
    if not os.access(output_dir, os.W_OK):
        print(f"[ERROR] No write permission for output directory: {output_dir}")
    else:
        print(f"[DEBUG] Write permission confirmed for: {output_dir}")

    # Initialize robot in default pose only once
    print("[DEBUG] Initializing robot pose...")
    root_state = robot.data.default_root_state.clone()
    root_state[:, :3] += origins
    robot.write_root_pose_to_sim(root_state[:, :7])
    robot.write_root_velocity_to_sim(root_state[:, 7:])
    
    # Set to default joint position with no randomization
    joint_pos, joint_vel = robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone()
    robot.write_joint_state_to_sim(joint_pos, joint_vel)
    robot.reset()
    print("[DEBUG] Robot initialized successfully")

    # Ensure we've created the directory for saving tool tip images specifically
    tool_tip_dir = os.path.join(output_dir, "tool_tip_images")
    if not os.path.exists(tool_tip_dir):
        os.makedirs(tool_tip_dir)
        print(f"[DEBUG] Created tool tip images directory: {tool_tip_dir}")

    img_count = 0
    while simulation_app.is_running():
        if count % 100 == 0:
            print(f"[DEBUG] Processing frame {count}...")
            
            try:
                if 'rgb' in cam.data.output:
                    print(f"[DEBUG] Image shape: {cam.data.output['rgb'].shape}")
                    # Convert tensor to numpy array
                    rgb_tensor = cam.data.output['rgb']
                    # Convert to numpy and ensure it's uint8 format (0-255)
                    rgb_np = rgb_tensor[0].detach().cpu().numpy()  # Remove batch dimension
                    
                    # Ensure RGB values are in the correct range (0-255)
                    if rgb_np.max() <= 1.0:
                        rgb_np = (rgb_np * 255).astype(np.uint8)
                    else:
                        rgb_np = rgb_np.astype(np.uint8)
                    
                    # Save image directly with PIL
                    img_filename = os.path.join(tool_tip_dir, f"tool_tip_{img_count:04d}.png")
                    img = Image.fromarray(rgb_np)
                    img.save(img_filename)
                    print(f"[DEBUG] Image saved to: {img_filename}")
                    img_count += 1
                else:
                    print(f"[ERROR] RGB data not available in camera output")
            except Exception as e:
                print(f"[ERROR] Error saving image: {e}")
                import traceback
                print(traceback.format_exc())

        # Apply zero efforts to keep the robot in place
        efforts = torch.zeros_like(robot.data.joint_pos)
        robot.set_joint_effort_target(efforts)
        robot.write_data_to_sim()
        sim.step()
        count += 1
        robot.update(sim_dt)

        if count == 1000:
            print("[DEBUG] Reached maximum frame count (5000), exiting...")
            break

def main():
    print("[DEBUG] Starting application...")
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    print(f"[DEBUG] Created simulation context with device: {sim.device}")
    
    sim.set_camera_view([2.5, 0.0, 4.0], [0.0, 0.0, 2.0])
    print("[DEBUG] Set simulation camera view")
    
    scene_entities, scene_origins = design_scene()
    scene_origins = torch.tensor(scene_origins, device=sim.device)
    
    sim.reset()
    print("[INFO]: Setup complete...")
    
    run_simulator(sim, scene_entities, scene_origins)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] Exception in main thread: {e}")
        import traceback
        print(traceback.format_exc())
    finally:
        print("[DEBUG] Closing simulation app...")
        simulation_app.close()
        print("[DEBUG] Simulation app closed")