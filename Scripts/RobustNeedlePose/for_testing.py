import numpy as np
import pyvista as pv

points = np.load("/home/imad/SurgicalToolsPose/TestImages/_out_sdrec/Left/pointcloud/pointcloud_0000.npy")
colors_rgba = np.load("/home/imad/SurgicalToolsPose/TestImages/_out_sdrec/Left/pointcloud/pointcloud_rgb_0000.npy")

assert points.shape[0] == colors_rgba.shape[0], "Mismatch in number of points and colors"

colors_rgb = colors_rgba[:, :3].astype(np.uint8)

cloud = pv.PolyData(points)
cloud.point_data['colors'] = colors_rgb

plotter = pv.Plotter()
plotter.add_mesh(cloud, scalars='colors', rgb=True, point_size=2, render_points_as_spheres=True)
plotter.show()
