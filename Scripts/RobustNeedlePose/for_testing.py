import imageio.v2 as imageio
import glob
import os

image_folder = '/home/imad/SurgicalToolsPose/'
image_files = sorted(glob.glob(os.path.join(image_folder, 'needle_tracking_frame_*.png')))

images = [imageio.imread(img) for img in image_files]
imageio.mimsave('output.gif', images, fps=30)
