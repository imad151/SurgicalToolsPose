import argparse
from omni.isaac.lab.app import AppLauncher

parser = argparse.ArgumentParser(description="idk man last choice")
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
    cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Lights", cfg)

    origins = [[0.0, 0.0, 0.0]]
    prim_utils.create_prim("/World/Origin", "Xform", translation=origins[0])

    psm_cfg = PSM_HIGH_PD_CFG.copy()
    psm_cfg.prim_path = "/World/Origin/Robot"
    psm = Articulation(psm_cfg)

    cam: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/Camera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(-0.34585, 0.4165, -0.20154), 
            rot=R.from_euler('xyz', [75, -180, 40], degrees=True).as_quat(), 
            convention='world'
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1e5)
        ),
        height=480,
        width=640
    )

    cam = TiledCamera(cam)
    scene_entities = {"psm": psm,
                      "cam": cam}
    return scene_entities, origins

def run_simulator(sim: sim_utils.SimulationContext, entities: dict[str, Articulation], origins: torch.Tensor):
    robot = entities["psm"]
    cam = entities["cam"]
    sim_dt = sim.get_physics_dt()
    count = 0

    output_dir = r"/home/imad/SurgicalToolsPose/TestImages/"
    rep_writer = rep.BasicWriter(
        output_dir=output_dir,
        frame_padding=0
    )

    while simulation_app.is_running():
        if count % 500 == 0:
            root_state = robot.data.default_root_state.clone()
            root_state[:, :3] += origins
            robot.write_root_pose_to_sim(root_state[:, :7])
            robot.write_root_velocity_to_sim(root_state[:, 7:])
            joint_pos, joint_vel = robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone()
            joint_pos += torch.rand_like(joint_pos) * 0.1
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            robot.reset()

        
            print(f"image shape: {cam.data.output['rgb'].shape}")
            single_cam_data = convert_dict_to_backend(
                {k: v for k, v in cam.data.output.items()}, backend="numpy"
            )
            single_cam_info = cam.data.info

            rep_output = {"annotators": {}}
            for key, data, info in zip(single_cam_data.keys(), single_cam_data.values(), single_cam_info.values()):
                if info is not None:
                    rep_output["annotators"][key] = {"render_product": {"data": data, **info}}
                else:
                    rep_output["annotators"][key] = {"render_product": {"data": data}}
            rep_output["trigger_outputs"] = {"on_time": cam.frame}
            rep_writer.write(rep_output)


        efforts = torch.randn_like(robot.data.joint_pos) * 5.0
        robot.set_joint_effort_target(efforts)
        robot.write_data_to_sim()
        sim.step()
        count += 1
        robot.update(sim_dt)

        if count == 5000:
            break

def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view([2.5, 0.0, 4.0], [0.0, 0.0, 2.0])
    scene_entities, scene_origins = design_scene()
    scene_origins = torch.tensor(scene_origins, device=sim.device)
    sim.reset()
    print("[INFO]: Setup complete...")
    run_simulator(sim, scene_entities, scene_origins)

if __name__ == "__main__":
    main()
    simulation_app.close()