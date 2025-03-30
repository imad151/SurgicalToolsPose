import numpy as np
from scipy.optimize import least_squares
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Camera intrinsic parameters random for now
focal_length = 1.0
principal_point = np.array([320, 240])
camera_intrinsic_matrix = np.array([
    [focal_length, 0, principal_point[0]],
    [0, focal_length, principal_point[1]],
    [0, 0, 1]
])

def fit_plane_pca(points):
    centroid = np.mean(points, axis=0)
    pca = PCA(n_components=3)
    pca.fit(points - centroid)
    normal = pca.components_[-1]
    return centroid, normal, pca.components_[:-1]

def project_to_plane(points, centroid, plane_basis):
    projected = (points - centroid) @ plane_basis.T
    return projected[:, 0], projected[:, 1]

def ellipse_residuals(params, u, v, aspect_ratio):
    u_c, v_c, b, theta = params
    a = aspect_ratio * b
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    u_rot = (u - u_c) * cos_t + (v - v_c) * sin_t
    v_rot = -(u - u_c) * sin_t + (v - v_c) * cos_t
    return (u_rot**2 / a**2 + v_rot**2 / b**2 - 1)

def fit_ellipse(u, v, aspect_ratio):
    u_c, v_c = np.mean(u), np.mean(v)
    b_init = (np.max(v) - np.min(v)) / 2
    theta_init = 0
    params_init = [u_c, v_c, b_init, theta_init]
    result = least_squares(ellipse_residuals, params_init, args=(u, v, aspect_ratio))
    return result.x

def compute_yaw_pitch_roll(normal, v1, v2):
    yaw = np.arctan2(normal[1], normal[0])
    pitch = np.arctan2(-normal[2], np.sqrt(normal[0]**2 + normal[1]**2))
    roll = np.arctan2(v2[2], v1[2])
    return np.degrees([yaw, pitch, roll])

def estimate_needle_pose(points, aspect_ratio):
    centroid, normal, plane_basis = fit_plane_pca(points)
    u, v = project_to_plane(points, centroid, plane_basis)
    u_c, v_c, b, theta = fit_ellipse(u, v, aspect_ratio)
    ellipse_center = centroid + u_c * plane_basis[0] + v_c * plane_basis[1]
    yaw, pitch, roll = compute_yaw_pitch_roll(normal, plane_basis[0], plane_basis[1])
    return ellipse_center, (yaw, pitch, roll)

def apply_transformation(points, rotation_matrix, translation_vector):
    return points @ rotation_matrix.T + translation_vector

def project_to_camera(points, camera_intrinsic_matrix, camera_position, camera_orientation):
    rotation_matrix = np.array([
        [np.cos(np.radians(camera_orientation[0])), -np.sin(np.radians(camera_orientation[0])), 0],
        [np.sin(np.radians(camera_orientation[0])), np.cos(np.radians(camera_orientation[0])), 0],
        [0, 0, 1]
    ])
    transformed_points = apply_transformation(points, rotation_matrix, camera_position)
    image_points = []
    for point in transformed_points:
        homogeneous_point = np.append(point, 1)
        projected_point = camera_intrinsic_matrix @ point
        if point[2] != 0:
            projected_point = projected_point / point[2]
        image_points.append(projected_point[:2])
    return np.array(image_points)

def visualize_3d(points, position, orientation, a, b, camera_position, camera_orientation):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='b', marker='o', s=10, label='Transformed Points')
    ax.scatter(position[0], position[1], position[2], c='r', marker='^', s=100, label='Estimated Ellipse Center')

    yaw, pitch, roll = orientation
    rotation_matrix = np.array([
        [np.cos(np.radians(yaw)), -np.sin(np.radians(yaw)), 0],
        [np.sin(np.radians(yaw)), np.cos(np.radians(yaw)), 0],
        [0, 0, 1]
    ])

    arrow_length = 1.5
    ax.quiver(position[0], position[1], position[2], rotation_matrix[0, 0], rotation_matrix[0, 1], rotation_matrix[0, 2], length=arrow_length, color='g', label='Yaw')
    ax.quiver(position[0], position[1], position[2], rotation_matrix[1, 0], rotation_matrix[1, 1], rotation_matrix[1, 2], length=arrow_length, color='y', label='Pitch')
    ax.quiver(position[0], position[1], position[2], rotation_matrix[2, 0], rotation_matrix[2, 1], rotation_matrix[2, 2], length=arrow_length, color='c', label='Roll')

    origin = np.array([0, 0, 0])
    ax.scatter(origin[0], origin[1], origin[2], c='k', marker='o', s=100, label='Origin')

    theta_vals = np.linspace(0, 2 * np.pi, 100)
    ellipse_points = np.array([a * np.cos(theta_vals), b * np.sin(theta_vals), np.zeros_like(theta_vals)])
    ellipse_rotated = rotation_matrix @ ellipse_points
    ellipse_transformed = ellipse_rotated.T + position

    ax.plot(ellipse_transformed[:, 0], ellipse_transformed[:, 1], ellipse_transformed[:, 2], c='m', label='Fitted Ellipse')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Visualization of Ellipse Pose Estimation')
    ax.legend()

    image_points = project_to_camera(ellipse_transformed, camera_intrinsic_matrix, camera_position, camera_orientation)

    fig2 = plt.figure(figsize=(6, 6))
    plt.scatter(image_points[:, 0], image_points[:, 1], c='m', label='Projected Ellipse')
    plt.xlabel('Image X')
    plt.ylabel('Image Y')
    plt.title('2D Projection of Ellipse on Camera Image Plane')
    plt.legend()
    plt.show()

# Test setup
center = np.array([0, 0, 0])
normal = np.array([0, 0, 1])
a = 5.0  
b = 3.0  
aspect_ratio = a / b
theta = np.linspace(0, 2 * np.pi, 5)

points_ellipse = []
for t in theta:
    x = a * np.cos(t)
    y = b * np.sin(t)
    z = 0
    point = np.array([x, y, z])
    rotation_matrix = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
    point_rot = rotation_matrix @ point
    point_rot += center
    points_ellipse.append(point_rot)

points_ellipse = np.array(points_ellipse)

rotation_matrix = np.array([[np.cos(np.pi/4), -np.sin(np.pi/4), 0], 
                            [np.sin(np.pi/4), np.cos(np.pi/4), 0], 
                            [0, 0, 1]])
translation_vector = np.array([2, -3, 1])
transformed_points = apply_transformation(points_ellipse, rotation_matrix, translation_vector)

position, orientation = estimate_needle_pose(transformed_points, aspect_ratio)

print("Estimated Position:", np.round(position, 2))
print("Estimated Orientation (Yaw, Pitch, Roll):", np.round(orientation, 2))

camera_position = np.array([0, 0, 0])
camera_orientation = [0, 0, 0]

visualize_3d(transformed_points, position, orientation, a, b, camera_position, camera_orientation)
