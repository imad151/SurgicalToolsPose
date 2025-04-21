import numpy as np
import matplotlib.pyplot as plt

class EllipseTransformer:  
    @staticmethod
    def apply_transformation(points, rotation_matrix = None, translation_vector = None):
        """
        Apply a rigid transformation to a set of 3D points.
        
        Args:
            points (numpy.ndarray): Array of 3D points with shape (n, 3).
            rotation_matrix (numpy.ndarray): 3x3 rotation matrix.
            translation_vector (numpy.ndarray): 3D translation vector.
            
        Returns:
            numpy.ndarray: Transformed points.
        """
        if rotation_matrix is not None and translation_vector is not None:
            return points @ rotation_matrix.T + translation_vector
        if rotation_matrix is not None:
            return points @ rotation_matrix.T
        if translation_vector is not None:
            return points + translation_vector
    
    @staticmethod
    def get_rotation_matrix(yaw, pitch, roll):
        """
        Compute a 3D rotation matrix from yaw, pitch, and roll angles.
        
        Args:
            yaw (float): Yaw angle in radians.
            pitch (float): Pitch angle in radians.
            roll (float): Roll angle in radians.
            
        Returns:
            numpy.ndarray: 3x3 rotation matrix.
        """
        R_yaw = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])

        R_pitch = np.array([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ])

        R_roll = np.array([
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)]
        ])

        return R_yaw @ R_pitch @ R_roll


class EllipseVisualizer:
    @staticmethod
    def visualize_3d(position, orientation, a, b):
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
    
        yaw, pitch, roll = np.radians(orientation)
        
        rotation_matrix = EllipseTransformer.get_rotation_matrix(yaw, pitch, roll)
    
        theta_vals = np.linspace(0, 2 * np.pi, 100)
        ellipse_points = np.array([
            a * np.cos(theta_vals), 
            b * np.sin(theta_vals), 
            np.zeros_like(theta_vals)
        ])
    
        ellipse_rotated = rotation_matrix @ ellipse_points  # shape: (3, N)
        ellipse_transformed = ellipse_rotated.T + position  # shape: (N, 3)
    
        ax.plot(
            ellipse_transformed[:, 0], 
            ellipse_transformed[:, 1], 
            ellipse_transformed[:, 2], 
            c='g', linewidth=2, label='Fitted Ellipse'
        )
    
        ax.scatter(*position, color='red', s=60, label='Ellipse Center')
    
        axis_length = max(a, b) * 1.5
        axes_colors = ['r', 'g', 'b']
        axes_labels = ['X', 'Y', 'Z']
    
        for i in range(3):
            axis_vec = rotation_matrix[:, i] * axis_length
            ax.quiver(
                position[0], position[1], position[2],
                axis_vec[0], axis_vec[1], axis_vec[2],
                color=axes_colors[i],
                linewidth=2,
                arrow_length_ratio=0.1
            )
    
        # Global axis labels
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('3D Visualization of Fitted Ellipse & Orientation')
        ax.legend()
    
        ax.set_box_aspect([1, 1, 1])  # equal aspect ratio
    
        plt.tight_layout()
        plt.show()
    