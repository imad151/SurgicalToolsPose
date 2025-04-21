import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the 3 vectors as rows of a matrix
vectors = np.array([
    [0.2, -0.2, 1.0],
    [0.0,  1.0, 0.2],
    [1.0,  0.0, -0.2]
])

# Origin for all vectors
origin = np.zeros((3,))

# Set up 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot each vector
colors = ['r', 'g', 'b']
for i in range(3):
    ax.quiver(*origin, *vectors[i], color=colors[i], label=f'v{i+1}')

# Set axis limits for better visibility
ax.set_xlim([-1, 1.5])
ax.set_ylim([-1, 1.5])
ax.set_zlim([-1, 1.5])

# Label axes
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Vector Plot')
ax.legend()
plt.tight_layout()
plt.show()
