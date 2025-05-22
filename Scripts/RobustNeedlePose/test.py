import warnings ; warnings.filterwarnings("ignore")
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R
from matplotlib.animation import FuncAnimation
import yaml
import time

from main import EnhancedEllipsePoseEstimator

def read_config(dir_path):
    with open(dir_path, 'r') as f:
        config = yaml.unsafe_load(f)
    return config

class SyntheticDataGeneration:
    def __init__(self, config):
        self.sim_config = config['SyntheticNeedleSimulation']
        self.vis_config = config.get('visualization', {})
        self.enabled_vis = self.vis_config.get('enabled', True)
        self.save_animation = self.vis_config.get('save_animation', False)
        self.frame_stride = self.vis_config.get('frame_stride', 1)
        
        self.pose_estimator = EnhancedEllipsePoseEstimator(
            aspect_ratio=self.sim_config.get('aspect_ratio', 2.0),
            surgical_constraints=config.get('surgical_constraints', {}),
            ransac_params=config.get('RobustPlaneFitter', {}),
            ellipse_method=config.get('FitzgibbonEllipseFitter', {}).get('method', 'hybrid'),
            kalman_config=config.get('AdaptiveKalmanFiltering', {})
        )

        self.needle_length = self.sim_config['needle_length']
        self.needle_radius = self.sim_config['needle_radius']
        
        self.base_points = self.generate_points()

    def generate_points(self):
        needle_length = self.sim_config['needle_length']
        needle_radius = self.sim_config['needle_radius']
        aspect_ratio = self.sim_config['aspect_ratio']
        noise_level = self.sim_config['noise_level']
        num_points = self.sim_config['num_points']
        outlier_ratio = self.sim_config.get('outlier_ratio', 0.0)

        a = needle_length / 2
        b = needle_radius

        t = np.linspace(0, np.pi, num_points)
        
        x = a * np.cos(t)
        y = b * np.sin(t)
        z = np.zeros_like(x)

        if noise_level > 0:
            x += np.random.normal(0, noise_level, size=x.shape)
            y += np.random.normal(0, noise_level, size=y.shape)
            z += np.random.normal(0, noise_level, size=z.shape)
        
        points = np.stack((x, y, z), axis=1)
        
        if outlier_ratio > 0 and num_points > 0:
            n_outliers = int(outlier_ratio * num_points)
            if n_outliers > 0:
                outlier_range = max(needle_length, needle_radius) * 2
                outliers = np.random.uniform(-outlier_range, outlier_range, size=(n_outliers, 3))
                points = np.vstack([points, outliers])
                
        return points

    def generate_needle_arc(self, position, rotation, num_arc_points=50):
        """Generate needle arc points based on position and orientation"""
        a = self.needle_length / 2
        b = self.needle_radius
        
        # Generate arc in local coordinate system (half ellipse)
        t = np.linspace(0, np.pi, num_arc_points)
        x_local = a * np.cos(t)
        y_local = b * np.sin(t)
        z_local = np.zeros_like(x_local)
        
        # Stack points
        local_points = np.stack((x_local, y_local, z_local), axis=1)
        
        # Transform to world coordinates
        world_points = (rotation @ local_points.T).T + position
        
        return world_points

    def generate_trajectory(self, points):
        trajectory_config = self.sim_config['trajectory']

        duration = trajectory_config['duration']
        dt = trajectory_config['dt']

        pos_amp = np.array(trajectory_config['position_amplitude'])
        pos_freq = np.array(trajectory_config['position_frequency'])
        pos_drift = np.array(trajectory_config['position_drift'])
        
        orient_amp = np.array(trajectory_config.get('orientation_amplitude', [0.3, 0.2, 0.3]))
        orient_freq = np.array(trajectory_config.get('orientation_frequency', [0.05, 0.1, 0.1]))
        orient_drift = np.array(trajectory_config.get('orientation_drift', [0.005, 0.01, 0.01]))

        times = np.arange(0, duration, dt)
        trajectories = []
        ground_truth_poses = []

        for t in times:
            pos = pos_amp * np.sin(2 * np.pi * pos_freq * t) + pos_drift * t
            
            orient = orient_amp * np.sin(2 * np.pi * orient_freq * t) + orient_drift * t
            
            rot_matrix = R.from_euler('xyz', orient).as_matrix()
            
            transformed_points = (rot_matrix @ points.T).T + pos
            
            trajectories.append(transformed_points)
            ground_truth_poses.append((pos, rot_matrix))

        return np.array(trajectories), ground_truth_poses, times


    def run_solver(self, trajectories, times):
        estimated_positions = []
        estimated_rotations = []
        metrics_list = []
        time_list = []
        for i, (points, t) in enumerate(zip(trajectories, times)):
            
            start_time = time.time()
            position, rotation, metrics = self.pose_estimator.estimate_pose(points, timestamp=t)
            end_time = time.time()

            time_list.append(end_time - start_time)
            estimated_positions.append(position)
            estimated_rotations.append(rotation)
            metrics_list.append(metrics)
            
            if i % 10 == 0:
                print(f"Processing frame {i}/{len(trajectories)}, Average Time = {sum(time_list)/len(time_list):2f}s")
        
        return np.array(estimated_positions), np.array(estimated_rotations), metrics_list, time_list

    def visualize_results(self, trajectories, ground_truth_poses, estimated_positions, estimated_rotations, time_list, times):
        if not self.enabled_vis:
            return
            
        fig = plt.figure(figsize=(18, 10))
        
        ax1 = fig.add_subplot(221)
        ax1.set_title('Position Error (mm)')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Error (mm)')
        
        ax2 = fig.add_subplot(222)
        ax2.set_title('Orientation Error (degrees)')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Error (degrees)')
        
        ax3 = fig.add_subplot(223, projection='3d')
        ax3.set_title('Trajectory')
        ax3.set_xlabel('X (mm)')
        ax3.set_ylabel('Y (mm)')
        ax3.set_zlabel('Z (mm)')

        ax4 = fig.add_subplot(224)
        ax4.set_title("Processing Time")
        ax4.set_xlabel("Time(s)")
        ax4.set_ylabel("Processing Time (ms)")
        
        pos_errors = []
        rot_errors = []
        
        for i, (gt_pose, est_pos, est_rot) in enumerate(zip(ground_truth_poses, estimated_positions, estimated_rotations)):
            gt_pos, gt_rot = gt_pose
            
            pos_error = np.linalg.norm(gt_pos - est_pos)
            pos_errors.append(pos_error)
            
            rot_error_matrix = np.abs(gt_rot - est_rot)
            rot_error = np.rad2deg(np.linalg.norm(rot_error_matrix, 'fro') / (2 * np.sqrt(2)))
            rot_errors.append(rot_error)
        
        ax1.plot(times, pos_errors)
        ax1.grid(True)
        
        ax2.plot(times, rot_errors)
        ax2.grid(True)
        
        gt_positions = np.array([pos for pos, _ in ground_truth_poses])
        ax3.plot(gt_positions[:, 0], gt_positions[:, 1], gt_positions[:, 2], 'b-', label='Ground Truth')
        ax3.plot(estimated_positions[:, 0], estimated_positions[:, 1], estimated_positions[:, 2], 'r--', label='Estimated')
        ax3.legend()
        
        ax4.plot(times, time_list)
        ax4.grid(True)

        plt.tight_layout()
        
        if self.save_animation:
            self._create_animation(trajectories, ground_truth_poses, estimated_positions, estimated_rotations)
        
        plt.show()
    
    def _create_animation(self, trajectories, ground_truth_poses, estimated_positions, estimated_rotations):
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        max_range = 0
        for points in trajectories:
            max_val = np.max(np.abs(points))
            if max_val > max_range:
                max_range = max_val
        
        ax.set_xlim(-max_range, max_range)
        ax.set_ylim(-max_range, max_range)
        ax.set_zlim(-max_range, max_range)
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')
        ax.set_title('Needle Tracking Animation')
        
        points_plot, = ax.plot([], [], [], 'bo', markersize=4, label='Points')

        gt_center_plot, = ax.plot([], [], [], 'go', markersize=8, label='GT Center')
        
        est_center_plot, = ax.plot([], [], [], 'ro', markersize=8, label='Est Center')
        
        gt_traj_plot, = ax.plot([], [], [], 'g-', linewidth=1, label='GT Trajectory')
        est_traj_plot, = ax.plot([], [], [], 'r-', linewidth=1, label='Est Trajectory')

        # Ground truth needle arc
        gt_needle_plot, = ax.plot([], [], [], 'g-', linewidth=3, alpha=0.7, label="GT Needle Arc")
        
        # Estimated needle arc
        est_needle_plot, = ax.plot([], [], [], 'r-', linewidth=3, alpha=0.7, label="Est Needle Arc")
        
        frame_text = ax.text2D(0.02, 0.95, "", transform=ax.transAxes)
        
        axis_scale = max_range * 0.2
        
        gt_axes = None
        est_axes = None
        
        ax.legend()
        
        def init():
            points_plot.set_data([], [])
            points_plot.set_3d_properties([])
            
            gt_center_plot.set_data([], [])
            gt_center_plot.set_3d_properties([])
            
            est_center_plot.set_data([], [])
            est_center_plot.set_3d_properties([])
            
            gt_traj_plot.set_data([], [])
            gt_traj_plot.set_3d_properties([])
            
            est_traj_plot.set_data([], [])
            est_traj_plot.set_3d_properties([])

            gt_needle_plot.set_data([], [])
            gt_needle_plot.set_3d_properties([])
            
            est_needle_plot.set_data([], [])
            est_needle_plot.set_3d_properties([])
            
            frame_text.set_text("")
            
            return (points_plot, gt_center_plot, est_center_plot, gt_traj_plot, est_traj_plot, 
                   gt_needle_plot, est_needle_plot, frame_text)
        
        def update(frame_idx):
            nonlocal gt_axes, est_axes
            
            i = frame_idx * self.frame_stride
            if i >= len(trajectories):
                i = len(trajectories) - 1
            
            points = trajectories[i]
            x, y, z = points[:, 0], points[:, 1], points[:, 2]
            points_plot.set_data(x, y)
            points_plot.set_3d_properties(z)
            
            gt_pos, gt_rot = ground_truth_poses[i]
            gt_center_plot.set_data([gt_pos[0]], [gt_pos[1]])
            gt_center_plot.set_3d_properties([gt_pos[2]])

            # Generate and plot ground truth needle arc
            gt_needle_arc = self.generate_needle_arc(gt_pos, gt_rot)
            gt_needle_plot.set_data(gt_needle_arc[:, 0], gt_needle_arc[:, 1])
            gt_needle_plot.set_3d_properties(gt_needle_arc[:, 2])
            
            if gt_axes is not None:
                for artist in gt_axes:
                    if artist:
                        try:
                            artist.remove()
                        except:
                            pass
            
            gt_x_arrow = ax.quiver(gt_pos[0], gt_pos[1], gt_pos[2], 
                                  gt_rot[0, 0]*axis_scale, gt_rot[1, 0]*axis_scale, gt_rot[2, 0]*axis_scale, 
                                  color='r')
            gt_y_arrow = ax.quiver(gt_pos[0], gt_pos[1], gt_pos[2], 
                                  gt_rot[0, 1]*axis_scale, gt_rot[1, 1]*axis_scale, gt_rot[2, 1]*axis_scale, 
                                  color='g')
            gt_z_arrow = ax.quiver(gt_pos[0], gt_pos[1], gt_pos[2], 
                                  gt_rot[0, 2]*axis_scale, gt_rot[1, 2]*axis_scale, gt_rot[2, 2]*axis_scale, 
                                  color='b')
            gt_axes = [gt_x_arrow, gt_y_arrow, gt_z_arrow]
            
            est_pos = estimated_positions[i]
            est_rot = estimated_rotations[i]
            est_center_plot.set_data([est_pos[0]], [est_pos[1]])
            est_center_plot.set_3d_properties([est_pos[2]])
            
            # Generate and plot estimated needle arc
            est_needle_arc = self.generate_needle_arc(est_pos, est_rot)
            est_needle_plot.set_data(est_needle_arc[:, 0], est_needle_arc[:, 1])
            est_needle_plot.set_3d_properties(est_needle_arc[:, 2])
            
            if est_axes is not None:
                for artist in est_axes:
                    if artist:
                        try:
                            artist.remove()
                        except:
                            pass
            
            est_x_arrow = ax.quiver(est_pos[0], est_pos[1], est_pos[2], 
                                   est_rot[0, 0]*axis_scale, est_rot[1, 0]*axis_scale, est_rot[2, 0]*axis_scale, 
                                   color='r', linestyle='--')
            est_y_arrow = ax.quiver(est_pos[0], est_pos[1], est_pos[2], 
                                   est_rot[0, 1]*axis_scale, est_rot[1, 1]*axis_scale, est_rot[2, 1]*axis_scale, 
                                   color='g', linestyle='--')
            est_z_arrow = ax.quiver(est_pos[0], est_pos[1], est_pos[2], 
                                   est_rot[0, 2]*axis_scale, est_rot[1, 2]*axis_scale, est_rot[2, 2]*axis_scale, 
                                   color='b', linestyle='--')
            est_axes = [est_x_arrow, est_y_arrow, est_z_arrow]
            
            gt_traj = np.array([pose[0] for pose in ground_truth_poses[:i+1]])
            gt_traj_plot.set_data(gt_traj[:, 0], gt_traj[:, 1])
            gt_traj_plot.set_3d_properties(gt_traj[:, 2])
            
            est_traj = estimated_positions[:i+1]
            est_traj_plot.set_data(est_traj[:, 0], est_traj[:, 1])
            est_traj_plot.set_3d_properties(est_traj[:, 2])
            
            pos_error = np.linalg.norm(gt_pos - est_pos)
            rot_error_matrix = np.abs(gt_rot - est_rot)
            rot_error = np.rad2deg(np.linalg.norm(rot_error_matrix, 'fro') / (2 * np.sqrt(2)))
            
            frame_text.set_text(f'Frame: {i}\nPosition Error: {pos_error:.2f}mm\nOrientation Error: {rot_error:.2f}°')
            
            return (points_plot, gt_center_plot, est_center_plot, gt_traj_plot, est_traj_plot, 
                   gt_needle_plot, est_needle_plot, frame_text)
        
        try:
            num_frames = len(trajectories) // self.frame_stride
            if num_frames <= 0:
                num_frames = 1
                
            ani = FuncAnimation(fig, update, frames=num_frames, init_func=init, blit=False, interval=50)
            try:
                ani.save('needle_tracking.mp4', writer='ffmpeg', fps=20, dpi=100)
                print("Animation saved as 'needle_tracking.mp4'")
            except:
                print("ffmpeg not available. Attempting to save with pillow...")
                ani.save('needle_tracking.gif', writer='pillow', fps=10, dpi=80)
                print("Animation saved as 'needle_tracking.gif'")
        except Exception as e:
            print(f"Error creating animation: {e}")
        
        plt.close(fig)


def main():
    cfg_path = 'Scripts/RobustNeedlePose/config.yaml'
    try:
        cfg = read_config(cfg_path)
    except FileNotFoundError:
        print(f"Config file not found at {cfg_path}.")
        return
    
    print("Config loaded from: ", cfg_path)
    
    sim = SyntheticDataGeneration(cfg)
    
    print("Generating synthetic needle points...")
    points = sim.generate_points()
    print(f"Generated {len(points)} points")
    
    print("Generating trajectory...")
    trajectories, ground_truth_poses, times = sim.generate_trajectory(points)
    print(f"Generated trajectory with {len(trajectories)} frames")
    
    print("Running pose estimator on trajectory...")
    estimated_positions, estimated_rotations, metrics_list, time_list = sim.run_solver(trajectories, times)
    sim.metrics_list = metrics_list 
    
    pos_errors = [np.linalg.norm(gt_pos - est_pos) for (gt_pos, _), est_pos in zip(ground_truth_poses, estimated_positions)]
    mean_pos_error = np.mean(pos_errors)
    max_pos_error = np.max(pos_errors)
    
    rot_errors = []
    for (_, gt_rot), est_rot in zip(ground_truth_poses, estimated_rotations):
        rot_error_matrix = np.abs(gt_rot - est_rot)
        rot_error = np.rad2deg(np.linalg.norm(rot_error_matrix, 'fro') / (2 * np.sqrt(2)))
        rot_errors.append(rot_error)
    
    mean_rot_error = np.mean(rot_errors)
    max_rot_error = np.max(rot_errors)
    
    print("\nPose Estimation Results:")
    print(f"Mean Position Error: {mean_pos_error:.3f}mm")
    print(f"Max Position Error: {max_pos_error:.3f}mm")
    print(f"Mean Orientation Error: {mean_rot_error:.3f}°")
    print(f"Max Orientation Error: {max_rot_error:.3f}°")
    
    sim.visualize_results(trajectories, ground_truth_poses, estimated_positions, estimated_rotations, time_list, times)


if __name__ == "__main__":
    main()