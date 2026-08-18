import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def create_heart_3d(ax):
    t = np.linspace(-np.pi, np.pi, 100)
    x = 16 * np.sin(t)**3
    y = 13 * np.cos(t) - 5 * np.cos(2*t) - 2 * np.cos(3*t) - np.cos(4*t)
    z = np.zeros_like(t) # For a 2D heart initially, we'll make it 3D later

    # Create a 3D heart by rotating the 2D heart
    theta = np.linspace(0, 2 * np.pi, 50)
    X, Y, Z = [], [], []

    for angle in theta:
        rotation_matrix = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle),  np.cos(angle), 0],
            [0,              0,             1]
        ])
        for i in range(len(t)):
            rotated_point = np.dot(rotation_matrix, np.array([x[i], y[i], z[i]]))
            X.append(rotated_point[0])
            Y.append(rotated_point[1])
            Z.append(rotated_point[2])

    ax.scatter(X, Y, Z, c=np.random.rand(len(X)), cmap='magma', marker='o', s=10) # Using magma for purple-gold
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Beautiful Purple-Gold 3D Particle Heart')

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
create_heart_3d(ax)
plt.show()