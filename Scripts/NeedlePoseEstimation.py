import numpy as np
from scipy.optimize import least_squares
from sklearn.linear_model import RANSACRegressor
from sklearn.decomposition import PCA
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

from NeedleHelperFunctions import EllipseTransformer, EllipseVisualizer


class EllipsePoseEstimator:
    def __init__(self, aspect_ratio):
        """
        Initialize the ellipse pose estimator. Use method estimate_pose(3d points) to get pose
        Args:
            aspect_ratio (float): The aspect ratio of the ellipse (a/b).
        """
        self.aspect_ratio = aspect_ratio
        
    def fit_plane_pca(self, points):
        centroid = np.mean(points, axis=0)
        pca = PCA(n_components=3)
        pca.fit(points)
        normal = pca.components_[-1]
        return centroid, normal, pca.components_[:-1]
        
    def _project_to_plane(self, points, centroid, plane_basis):
        projected = (points - centroid) @ plane_basis.T
        return projected[:, 0], projected[:, 1]
        
    def _ellipse_residuals(self, params, u, v):
        u_c, v_c, b, theta = params
        a = self.aspect_ratio * b
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        u_rot = (u - u_c) * cos_t + (v - v_c) * sin_t
        v_rot = -(u - u_c) * sin_t + (v - v_c) * cos_t
        return (u_rot**2 / a**2 + v_rot**2 / b**2 - 1)
        
    def _fit_ellipse(self, u, v):
        u_c, v_c = np.mean(u), np.mean(v)
        b_init = (np.max(v) - np.min(v)) / 2
        theta_init = 0
        params_init = [u_c, v_c, b_init, theta_init]

        result = least_squares(
            self._ellipse_residuals, 
            params_init, 
            args=(u, v), 
            loss='huber', 
            f_scale=0.2
        )
        return result.x
        
    def _compute_yaw_pitch_roll(self, normal, v1, v2):
        rot_mat = np.column_stack((v1, v2, normal))

        if np.linalg.det(rot_mat) < 0:
            v2 = -v2 
            rot_mat = np.column_stack((v1, v2, normal))
        
        u, _, vh = np.linalg.svd(rot_mat)
        rot_mat = u @ vh

        return rot_mat
        
    def estimate_pose(self, points):
        """
        Estimate the pose of an ellipse from a set of 3D points.
        
        Args:
            points (numpy.ndarray): Array of 3D points with shape (n, 3).
            
        Returns:
            tuple: (position, orientation)
                position (numpy.ndarray): 3D position of the ellipse center.
                rot_mat (numpy.ndarray): rotation_mat from scipy.
        """
        centroid, normal, plane_basis = self.fit_plane_pca(points)
        u, v = self._project_to_plane(points, centroid, plane_basis)
        u_c, v_c, b, theta = self._fit_ellipse(u, v)
        ellipse_center = centroid + u_c * plane_basis[0] + v_c * plane_basis[1]
        rot_mat = self._compute_yaw_pitch_roll(normal, plane_basis[0], plane_basis[1])
        return ellipse_center, rot_mat


def generate_test_data(a, b, num_points=10):
    theta = np.linspace(0, np.pi, num_points)

    points_ellipse = []
    for t in theta:
        x = a * np.cos(t)
        y = b * np.sin(t)
        z = 0
        point = np.array([x, y, z])
        points_ellipse.append(point)
    
    points_array = np.array(points_ellipse)
    
    return points_array

def main():
    """Run a simple test of the ellipse pose estimator."""
    a = 1  # Semi-major axis
    b = 2  # Semi-minor axis
    aspect_ratio = a / b
    
    default_points = generate_test_data(a, b, num_points=8)
    points = EllipseTransformer.apply_transformation(default_points, None, np.array([0, -3, 1])) 

    estimator = EllipsePoseEstimator(aspect_ratio=aspect_ratio)
    pos, rot_mat = estimator.estimate_pose(points)
    print(np.round(pos, 2))
    print(np.round(R.from_matrix(rot_mat).as_euler("xyz", degrees=True), 2))


def test_plot():
    """
    Test the ellipse pose estimator with different numbers of points
    and visualize the results and errors.
    """
    a = 0.5
    b = 1.0
    aspect_ratio = a / b
    true_position = np.array([0, 0, 5])
    true_orientation = np.array([0, 10, 90])
    true_orientation = R.from_euler('xyz', true_orientation, degrees=True).as_matrix()
    print(np.round(true_orientation, 2))
    num_points_range = range(4, 21)
    position_errors = []
    orientation_errors = []

    for n in num_points_range:
        # Generate and transform test points
        default_points = generate_test_data(a, b, num_points=n)
        transformed_points = EllipseTransformer.apply_transformation(
            default_points, 
            true_orientation, 
            true_position
        )
        # Add some noise
        noise = np.random.uniform(low=-0.01, high=0.01, size=transformed_points.shape)
        noise -= noise.mean(axis=0)
        transformed_points += noise
        
        
        # Estimate pose
        estimator = EllipsePoseEstimator(aspect_ratio)
        estimated_position, estimated_orientation = estimator.estimate_pose(transformed_points)
        _, _, basis_current = estimator.fit_plane_pca(transformed_points)
        
        # Calculate errors
        pos_error = np.linalg.norm(estimated_position - true_position)
        ori_error = np.linalg.norm(estimated_orientation - true_orientation)

        position_errors.append(pos_error)
        orientation_errors.append(ori_error)

    # ---- Plot results ----
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    s, t = np.linspace(-5, 5, 10), np.linspace(-5, 5, 10)
    S, T = np.meshgrid(s, t)
    estimated_position = np.array([0, 0, 5])
    print(np.round(estimated_orientation, 2))
    plane_points = (
        estimated_position[:, np.newaxis, np.newaxis] + 
        S * basis_current[0][:, np.newaxis, np.newaxis] + 
        T * basis_current[1][:, np.newaxis, np.newaxis]
    )
    X, Y, Z = plane_points
    
    # Plot Plane
    ax.plot_surface(X, Y, Z, alpha=0.1, color='blue', edgecolor='none', antialiased=True)
    ax.quiver(*estimated_position, *basis_current[0], label='v1', color='red')
    ax.quiver(*estimated_position, *basis_current[1], label='v2', color='green')
    
    # Plot points
    for i in transformed_points:
        ax.scatter(i[0], i[1], i[2], c='red', marker='o')
    
    ax.legend()
    ax.view_init(elev=30, azim=90)
    ax.grid(False)
    
    # Error plots
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)

    plt.plot(num_points_range, position_errors, marker='o')
    plt.title('Position Error vs. Number of Keypoints')
    plt.xlabel('Number of Keypoints')
    plt.ylabel('Position Error ')
    plt.grid(False)

    plt.subplot(1, 2, 2)
    plt.plot(num_points_range, orientation_errors, marker='o', color='orange')
    plt.title('Orientation Error vs. Number of Keypoints')
    plt.xlabel('Number of Keypoints')
    plt.ylabel('Orientation Error ')
    plt.grid(False)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    test_plot()