import omni.replicator.core as rep
from omni.isaac.core.utils.stage import get_current_stage
from pxr import Usd, UsdGeom


import omni
import asyncio
import time

import numpy as np
import os
from pathlib import Path

# Configuration
CONFIG = {
    "output_dir": "/home/imad/SurgicalToolsPose/TestImages",
    "num_frames": 1,
    "needle_path": "/World/needle_sdf",
    "left_camera_path": "/World/Realsense/RSD455/Camera_OmniVision_OV9782_Left",
    "right_camera_path": "/World/Realsense/RSD455/Camera_OmniVision_OV9782_Right",
    "resolution": (1280, 800),
    "samples_per_pixel": 64,
    "rt_subframes": 40
}


def create_writer_and_rp(base_dir=None):
    """Create render products and writers for the cameras."""
    left_camera = CONFIG["left_camera_path"]
    right_camera = CONFIG["right_camera_path"]
    resolution = CONFIG["resolution"]
    
    rp1 = rep.create.render_product(left_camera, resolution=resolution)
    rp2 = rep.create.render_product(right_camera, resolution=resolution)

    output_dir = base_dir or CONFIG["output_dir"]
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    left_dir = output_path / "left"
    right_dir = output_path / "right"

    for dir_path in [left_dir, right_dir]:
        dir_path.mkdir(exist_ok=True)
    
    writers = []

    writer1 = rep.WriterRegistry.get("BasicWriter")
    writer1.initialize(
        rgb=True, 
        output_dir=str(left_dir)
    )
    writer1.attach([rp1])

    writer2 = rep.WriterRegistry.get("BasicWriter")
    writer2.initialize(
        rgb=True, 
        output_dir=str(right_dir)
    )
    writer2.attach([rp2])

    writers.append(writer1)
    writers.append(writer2)

    rep.orchestrator.set_capture_on_play(False)

    return writers

async def capture_img():
    await rep.orchestrator.step_async(rt_subframes=CONFIG["rt_subframes"])

def change_settings():
    omni.log.get_log().enabled = False
    rep.settings.set_render_pathtraced(samples_per_pixel=CONFIG["samples_per_pixel"])


def get_camera_to_object_transform(camera_path, object_path, time_code=None):
    stage = get_current_stage()

    camera_prim = stage.GetPrimAtPath(camera_path)
    object_prim = stage.GetPrimAtPath(object_path)

    time_code = Usd.TimeCode.Default()

    camera_xformable = UsdGeom.Xformable(camera_prim)
    object_xformable = UsdGeom.Xformable(object_prim)
    
    camera_world_matrix = camera_xformable.ComputeLocalToWorldTransform(time_code)
    object_world_matrix = object_xformable.ComputeLocalToWorldTransform(time_code)
    
    camera_world_np = np.array(camera_world_matrix).reshape(4, 4)
    object_world_np = np.array(object_world_matrix).reshape(4, 4)

    
    camera_world_inverse = np.linalg.inv(camera_world_np)
    
    camera_to_object = np.matmul(camera_world_inverse, object_world_np)
    
    return camera_to_object


def get_translation_between_prims(source_prim_path, target_prim_path):
    stage = omni.usd.get_context().get_stage()
    
    source_prim = stage.GetPrimAtPath(source_prim_path)
    target_prim = stage.GetPrimAtPath(target_prim_path)
        
    source_world_matrix = omni.usd.get_world_transform_matrix(source_prim)
    target_world_matrix = omni.usd.get_world_transform_matrix(target_prim)
    
    source_world_pos = np.array([source_world_matrix[3][0], 
                                 source_world_matrix[3][1], 
                                 source_world_matrix[3][2]])
    
    target_world_pos = np.array([target_world_matrix[3][0], 
                                 target_world_matrix[3][1], 
                                 target_world_matrix[3][2]])
    
    translation_vector = target_world_pos - source_world_pos
    
    return translation_vector

def save_transform_matrix(matrix, filepath, round_digits=2):
    rounded_matrix = np.round(matrix, round_digits)
    np.save(filepath, rounded_matrix)
    return rounded_matrix


async def main():
    transform_dir = create_writer_and_rp()
    change_settings()
    
    print(f"Starting data generation for {CONFIG['num_frames']} frames...")
    
    try:
        for i in range(CONFIG["num_frames"]):

            ##### Save Image #####                        
            await capture_img()

            ##### Save Keypoint Info #####
            for i in range(1, 11):
                left_to_keypoint_n = get_translation_between_prims(
                    CONFIG["left_camera_path"],
                    f"/keypoint_{i}"
                )
                if left_to_keypoint_n is not None:
                    left_transform_path = os.path.join(CONFIG["output_dir"], "keypoints")
                    os.makedirs(left_transform_path, exist_ok=True)
                    left_transform_path = os.path.join(left_transform_path, f"left_cam_to_keypoint_{i}.npy")
                    rounded_left_to_needle = save_transform_matrix(left_to_keypoint_n, left_transform_path)

            for i in range(1, 11):      
                right_to_keypoint_n = get_translation_between_prims(
                    CONFIG["right_camera_path"],
                    f"/keypoint_{i}"
                )
                if right_to_keypoint_n is not None:
                    right_transform_path = os.path.join(CONFIG["output_dir"], "keypoints")
                    os.makedirs(right_transform_path, exist_ok=True)
                    right_transform_path = os.path.join(right_transform_path, f"right_cam_to_keypoint_{i}.npy")
                    rounded_left_to_needle = save_transform_matrix(right_to_keypoint_n, left_transform_path)
            
            left_to_center = get_translation_between_prims(CONFIG["left_camera_path"], CONFIG["needle_path"])
            _ = save_transform_matrix(left_to_center, "/home/imad/SurgicalToolsPose/needle.npy")
            ##### Save Needle Orientation
            left_to_needle = get_camera_to_object_transform(
                CONFIG["left_camera_path"], 
                CONFIG["needle_path"]
            )
            right_to_needle = get_camera_to_object_transform(
                CONFIG["right_camera_path"], 
                CONFIG["needle_path"]
            )

            if left_to_needle is not None:
                left_transform_path = os.path.join(CONFIG["output_dir"], f"left_to_needle_visible_{i:06d}.npy")
                rounded_left_to_needle = save_transform_matrix(left_to_needle, left_transform_path)

        
    except Exception as e:
        print(f"Error during data generation: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        print(f"Data generation finished - output saved to {CONFIG['output_dir']}")


asyncio.ensure_future(main())
