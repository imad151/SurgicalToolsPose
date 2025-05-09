import warnings ; warnings.simplefilter("ignore")
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
from typing import Tuple, List, Dict, Optional

from main import EnhancedEllipsePoseEstimator

class SyntheticNeedleDataGenerator:
    """
    Generates synthetic 3D needle point clouds and visualizes tracking results.
    
    This class provides methods for:
    1. Generating realistic needle point clouds with configurable noise and outliers
    2. Creating needle trajectories with realistic surgical motion patterns
    3. Visualizing the tracking results and comparing with ground truth
    4. Evaluating tracking performance with quantitative metrics
    
    Attributes:
        needle_length (float): Length of the needle in mm.
        needle_radius (float): Radius of the needle in mm.
        aspect_ratio (float): Aspect ratio of the elliptical cross-section.
        noise_level (float): Standard deviation of Gaussian noise in mm.
        outlier_ratio (float): Ratio of outlier points.
        trajectory_params (dict): Parameters for trajectory generation.
    """
    
    def __init__(self, 
                 needle_length: float = 26.0,  # Standard 1-inch surgical needle
                 needle_radius: float = 13.0,  # Half-circle needle
                 aspect_ratio: float = 1.0,    # Circular cross-section by default
                 noise_level: float = 0.5,     # 0.5mm noise 
                 outlier_ratio: float = 0.1,   # 10% outliers
                 trajectory_params: Optional[dict] = None):
        """
        Initialize the synthetic needle data generator.
        
        Args:
            needle_length: Length of the needle in mm.
            needle_radius: Radius of the needle in mm.
            aspect_ratio: Aspect ratio of the elliptical cross-section.
            noise_level: Standard deviation of Gaussian noise in mm.
            outlier_ratio: Ratio of outlier points.
            trajectory_params: Parameters for trajectory generation.
        """
        self.needle_length = needle_length
        self.needle_radius = needle_radius
        self.aspect_ratio = aspect_ratio
        self.noise_level = noise_level
        self.outlier_ratio = outlier_ratio
        
        # Default trajectory params
        self.trajectory_params = trajectory_params or {
            'position_amplitude': [20.0, 15.0, 10.0],  # mm
            'position_frequency': [0.1, 0.15, 0.2],    # Hz
            'orientation_amplitude': [np.pi/4, np.pi/6, np.pi/3],  # radians
            'orientation_frequency': [0.05, 0.1, 0.15],  # Hz
            'position_drift': [0.2, 0.3, 0.1],  # mm/s
            'orientation_drift': [0.01, 0.02, 0.015]  # rad/s
        }
        
        self.gt_positions = []
        self.gt_orientations = []
        self.timestamps = []
        
        self.est_positions = []
        self.est_orientations = []
        
        self.metrics = {
            'position_error': [],
            'orientation_error': [],
            'processing_time': []
        }
    
    def generate_needle_points(self, 
                              position: np.ndarray, 
                              orientation: np.ndarray, 
                              n_points: int = 50) -> np.ndarray:
        """
        Generate synthetic 3D points lying on a needle.
        
        Args:
            position: 3D position of the needle center.
            orientation: 3x3 rotation matrix defining needle orientation.
            n_points: Number of points to generate.
            
        Returns:
            3D point cloud of shape (n_points, 3).
        """
        theta = np.linspace(0, np.pi, n_points)
        
        points = np.zeros((n_points, 3))
        points[:, 0] = self.needle_radius * np.cos(theta)
        points[:, 2] = self.needle_radius * np.sin(theta)
        
        if self.aspect_ratio != 1.0:
            points[:, 0] *= self.aspect_ratio
        
        points = np.dot(points, orientation.T) + position
        
        points += np.random.normal(0, self.noise_level, points.shape)
        
        n_outliers = int(n_points * self.outlier_ratio)
        if n_outliers > 0:
            outlier_indices = np.random.choice(n_points, n_outliers, replace=False)
            outlier_magnitude = self.needle_radius * 0.5
            points[outlier_indices] += np.random.uniform(-outlier_magnitude, outlier_magnitude, (n_outliers, 3))
        
        visible_ratio = np.random.uniform(0.7, 1.0)  # 70-100% visible
        visible_indices = np.random.choice(n_points, int(visible_ratio * n_points), replace=False)
        
        return points[visible_indices]
    
    def generate_trajectory(self, 
                           duration: float = 10.0, 
                           dt: float = 0.033) -> List[Tuple[np.ndarray, np.ndarray, float]]:
        """
        Generate a realistic needle trajectory.
        
        Args:
            duration: Duration of the trajectory in seconds.
            dt: Time step in seconds.
            
        Returns:
            List of (position, orientation, timestamp) tuples.
        """
        n_frames = int(duration / dt)
        
        trajectory = []
        
        position = np.zeros(3)
        orientation = np.eye(3)
        
        t = np.arange(n_frames) * dt
        
        for i in range(n_frames):
            # Position: sinusoidal motion + drift
            pos_amp = self.trajectory_params['position_amplitude']
            pos_freq = self.trajectory_params['position_frequency']
            pos_drift = self.trajectory_params['position_drift']
            
            position = np.array([
                pos_amp[0] * np.sin(2 * np.pi * pos_freq[0] * t[i]) + pos_drift[0] * t[i],
                pos_amp[1] * np.cos(2 * np.pi * pos_freq[1] * t[i]) + pos_drift[1] * t[i],
                pos_amp[2] * np.sin(2 * np.pi * pos_freq[2] * t[i]) + pos_drift[2] * t[i]
            ])
            
            # Orientation: sinusoidal rotation + drift
            ori_amp = self.trajectory_params['orientation_amplitude']
            ori_freq = self.trajectory_params['orientation_frequency']
            ori_drift = self.trajectory_params['orientation_drift']
            
            yaw = ori_amp[0] * np.sin(2 * np.pi * ori_freq[0] * t[i]) + ori_drift[0] * t[i]
            pitch = ori_amp[1] * np.cos(2 * np.pi * ori_freq[1] * t[i]) + ori_drift[1] * t[i]
            roll = ori_amp[2] * np.sin(2 * np.pi * ori_freq[2] * t[i]) + ori_drift[2] * t[i]
            
            Rz = np.array([
                [np.cos(yaw), -np.sin(yaw), 0],
                [np.sin(yaw), np.cos(yaw), 0],
                [0, 0, 1]
            ])
            
            Ry = np.array([
                [np.cos(pitch), 0, np.sin(pitch)],
                [0, 1, 0],
                [-np.sin(pitch), 0, np.cos(pitch)]
            ])
            
            Rx = np.array([
                [1, 0, 0],
                [0, np.cos(roll), -np.sin(roll)],
                [0, np.sin(roll), np.cos(roll)]
            ])
            
            orientation = Rz @ Ry @ Rx
            
            self.gt_positions.append(position.copy())
            self.gt_orientations.append(orientation.copy())
            self.timestamps.append(t[i])
            
            trajectory.append((position.copy(), orientation.copy(), t[i]))
        
        return trajectory
    
    def run_tracking_simulation(self, 
                               trajectory: List[Tuple[np.ndarray, np.ndarray, float]], 
                               visualize: bool = True,
                               save_animation: bool = False) -> Dict:
        """
        Run a tracking simulation using the enhanced ellipse pose estimator.
        
        Args:
            trajectory: List of (position, orientation, timestamp) tuples.
            visualize: Whether to visualize the tracking results.
            save_animation: Whether to save the animation as a video.
            
        Returns:
            Dictionary of tracking performance metrics.
        """
        surgical_constraints = {
            'needle_length_mm': self.needle_length,
            'needle_radius_mm': self.needle_radius,
            'max_translation_mm': [100, 100, 100],
            'max_rotation_deg': [360, 60, 360],
            'stereo_baseline': 5.0,
            'working_distance': 100.0
        }
        
        estimator = EnhancedEllipsePoseEstimator(
            aspect_ratio=self.aspect_ratio,
            surgical_constraints=surgical_constraints
        )
        
        if visualize:
            fig = plt.figure(figsize=(15, 10))
            ax = fig.add_subplot(111, projection='3d')
            plt.ion()  
        
        for i, (position, orientation, timestamp) in enumerate(trajectory):
            # Generate synthetic points
            points = self.generate_needle_points(position, orientation)
            
            # Estimate pose
            start_time = time.time()
            est_position, est_orientation, metrics = estimator.estimate_pose(points, timestamp)
            processing_time = time.time() - start_time
            
            # Store results
            self.est_positions.append(est_position)
            self.est_orientations.append(est_orientation)
            
            # Compute errors
            position_error = np.linalg.norm(est_position - position)
            orientation_error_rad = np.arccos(
                np.clip((np.trace(est_orientation @ orientation.T) - 1) / 2, -1, 1)
            )
            orientation_error_deg = np.degrees(orientation_error_rad)
            
            # Store metrics
            self.metrics['position_error'].append(position_error)
            self.metrics['orientation_error'].append(orientation_error_deg)
            self.metrics['processing_time'].append(processing_time)
            
            # Visualize
            if visualize and (i % 5 == 0 or i == len(trajectory) - 1):
                ax.clear()
                
                ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='b', s=10, label='Input Points')
                
                # Plot ground truth needle
                self._plot_needle(ax, position, orientation, color='g', label='Ground Truth')
                
                # Plot estimated needle
                self._plot_needle(ax, est_position, est_orientation, color='r', label='Estimated')
                
                max_range = max([
                    np.max(np.abs(points[:, 0])),
                    np.max(np.abs(points[:, 1])),
                    np.max(np.abs(points[:, 2]))
                ]) * 1.5
                
                ax.set_xlim(-max_range, max_range)
                ax.set_ylim(-max_range, max_range)
                ax.set_zlim(-max_range, max_range)
                
                ax.set_xlabel('X (mm)')
                ax.set_ylabel('Y (mm)')
                ax.set_zlabel('Z (mm)')
                ax.set_title(f'Frame {i}: Pos Error = {position_error:.2f}mm, Orient Error = {orientation_error_deg:.2f}°')
                ax.legend()
                
                plt.draw()
                plt.pause(0.001)
                
                if save_animation:
                    plt.savefig(f'needle_tracking_frame_{i:04d}.png')
        
        if visualize:
            plt.ioff()
            plt.show()
        
        summary = {
            'mean_position_error': np.mean(self.metrics['position_error']),
            'std_position_error': np.std(self.metrics['position_error']),
            'max_position_error': np.max(self.metrics['position_error']),
            'mean_orientation_error': np.mean(self.metrics['orientation_error']),
            'std_orientation_error': np.std(self.metrics['orientation_error']),
            'max_orientation_error': np.max(self.metrics['orientation_error']),
            'mean_processing_time': np.mean(self.metrics['processing_time']),
            'max_processing_time': np.max(self.metrics['processing_time'])
        }
        
        return summary
    
    def _plot_needle(self, 
                    ax, 
                    position: np.ndarray, 
                    orientation: np.ndarray, 
                    color: str = 'r',
                    label: str = 'Needle'):
        """
        Plot a needle as a half-circle in 3D.
        
        Args:
            ax: Matplotlib 3D axis.
            position: 3D position of the needle center.
            orientation: 3x3 rotation matrix.
            color: Color of the needle.
            label: Label for the legend.
        """
        theta = np.linspace(0, np.pi, 100)
        needle_points = np.zeros((100, 3))
        needle_points[:, 0] = self.needle_radius * np.cos(theta)
        needle_points[:, 2] = self.needle_radius * np.sin(theta)
        
        if self.aspect_ratio != 1.0:
            needle_points[:, 0] *= self.aspect_ratio
        
        needle_3d = np.dot(needle_points, orientation.T) + position
        
        ax.plot(needle_3d[:, 0], needle_3d[:, 1], needle_3d[:, 2], color=color, linewidth=2, label=label)
        
        ax.scatter(position[0], position[1], position[2], color=color, s=50)
        
        axis_length = self.needle_radius * 0.5
        for i, c in enumerate(['r', 'g', 'b']):
            if i == 0 and color == 'r':
                c = 'm'
            axis = np.zeros((2, 3))
            axis[1, i] = axis_length
            transformed_axis = np.dot(axis, orientation.T) + position
            ax.plot(transformed_axis[:, 0], transformed_axis[:, 1], transformed_axis[:, 2], 
                    color=c, linewidth=1.5)
    
    def plot_trajectory_comparison(self):
        """
        Plot a comparison of ground truth and estimated trajectories.
        """
        gt_positions = np.array(self.gt_positions)
        est_positions = np.array(self.est_positions)
        timestamps = np.array(self.timestamps)
        
        fig = plt.figure(figsize=(15, 10))
        
        # Position comparison
        ax1 = fig.add_subplot(221, projection='3d')
        ax1.plot(gt_positions[:, 0], gt_positions[:, 1], gt_positions[:, 2], 'g-', label='Ground Truth')
        ax1.plot(est_positions[:, 0], est_positions[:, 1], est_positions[:, 2], 'r-', label='Estimated')
        ax1.set_xlabel('X (mm)')
        ax1.set_ylabel('Y (mm)')
        ax1.set_zlabel('Z (mm)')
        ax1.set_title('3D Trajectory Comparison')
        ax1.legend()
        
        # Position error over time
        ax2 = fig.add_subplot(222)
        position_errors = [np.linalg.norm(est - gt) for est, gt in zip(est_positions, gt_positions)]
        ax2.plot(timestamps, position_errors, 'b-')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Position Error (mm)')
        ax2.set_title('Position Error Over Time')
        ax2.grid(True)
        
        # Orientation error over time
        ax3 = fig.add_subplot(223)
        ax3.plot(timestamps, self.metrics['orientation_error'], 'r-')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Orientation Error (degrees)')
        ax3.set_title('Orientation Error Over Time')
        ax3.grid(True)
        
        # Processing time
        ax4 = fig.add_subplot(224)
        ax4.plot(timestamps, self.metrics['processing_time'], 'g-')
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Processing Time (s)')
        ax4.set_title('Processing Time Per Frame')
        ax4.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def plot_error_histograms(self):
        """
        Plot histograms of position and orientation errors.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        ax1.hist(self.metrics['position_error'], bins=20, color='blue', alpha=0.7)
        ax1.set_xlabel('Position Error (mm)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Position Error Distribution')
        ax1.grid(True)
        
        ax2.hist(self.metrics['orientation_error'], bins=20, color='red', alpha=0.7)
        ax2.set_xlabel('Orientation Error (degrees)')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Orientation Error Distribution')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()


def main():
    generator = SyntheticNeedleDataGenerator(
        needle_length=26.0,  # 26mm needle (standard surgical needle)
        needle_radius=13.0,  # Half-circle needle
        aspect_ratio=1.0,    # Circular cross-section
        noise_level=0.5,     # 0.5mm noise
        outlier_ratio=0.1    # 10% outliers
    )
    
    # Generate trajectory (10 seconds at 30Hz)
    print("Generating synthetic needle trajectory...")
    trajectory = generator.generate_trajectory(duration=10.0, dt=0.033)
    
    # Run tracking simulation
    print("Running tracking simulation...")
    summary = generator.run_tracking_simulation(
        trajectory=trajectory,
        visualize=True,
        save_animation=True
    )
    
    # Print summary statistics
    print("\nTracking Performance Summary:")
    print(f"Mean Position Error: {summary['mean_position_error']:.2f} mm")
    print(f"Max Position Error: {summary['max_position_error']:.2f} mm")
    print(f"Mean Orientation Error: {summary['mean_orientation_error']:.2f} degrees")
    print(f"Max Orientation Error: {summary['max_orientation_error']:.2f} degrees")
    print(f"Mean Processing Time: {summary['mean_processing_time']*1000:.2f} ms")
    
    # Plot trajectory comparison
    generator.plot_trajectory_comparison()
    
    # Plot error histograms
    generator.plot_error_histograms()

if __name__ == "__main__":
    main()
