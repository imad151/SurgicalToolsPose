import os

import numpy as np
import matplotlib.pyplot as plt
import cv2
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.image as mpimg
from scipy.spatial.transform import Rotation as R

from NeedlePoseEstimation import EllipsePoseEstimator

def get_keypoints(dir_path: str) -> np.ndarray:
    if not os.path.exists(dir_path):
        return
    keypoints = None
    for filename in os.listdir(dir_path):
        if 'left' in filename:
            data = np.load(os.path.join(dir_path, filename))
            keypoints = data if keypoints is None else np.vstack((keypoints, data))
    return keypoints


def get_pose(keypoints: np.ndarray)->tuple[np.ndarray, np.ndarray]:
    est = EllipsePoseEstimator(1.0)
    _, normal, plane_basis = est.fit_plane_pca(keypoints)
    pos, orient = est.estimate_pose(keypoints)

    return pos, orient, normal, plane_basis


points = get_keypoints("/home/imad/SurgicalToolsPose/TestImages/keypoints/")
pos, orient, normal, plane_basis = get_pose(points)
print(orient)
print(np.load("/home/imad/SurgicalToolsPose/TestImages/left_to_needle_visible_000010.npy"))
plane_size = 1
grid_range = np.linspace(-plane_size, plane_size, 10)
u, v = np.meshgrid(grid_range, grid_range)

plane_pts = pos[:, None, None] + u * plane_basis[0][:, None, None] + v * plane_basis[1][:, None, None]
X, Y, Z = plane_pts[0], plane_pts[1], plane_pts[2]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for point in points:
    ax.scatter(point[0], point[1], point[2], marker='o')

ax.plot_surface(X, Y, Z, color='cyan', alpha=0.5)

ax.grid(False)
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_zlim(-1, 1)
plt.show()