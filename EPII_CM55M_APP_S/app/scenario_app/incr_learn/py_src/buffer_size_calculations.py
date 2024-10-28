import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Size of example in bytes
size_per_example = 28 * 28 + 1

N_RAM = np.arange(100, 2000, 100)
N_EEPROM = np.arange(100, 1000, 100)

# Size of RAM and EEPROM buffers in MegaBytes
RAM_buffer_size = N_RAM * size_per_example / (1024 * 1024)
EEPROM_buffer_size = N_EEPROM * size_per_example / (1024 * 1024)

total_number_of_examples = N_RAM[:, np.newaxis] + N_EEPROM

# distance matrix size in MegaBytes (assume 2 bytes per Xdistance value)
dist_matrix_size = 2 * np.square(total_number_of_examples) / (1024 * 1024)

X, Y = np.meshgrid(RAM_buffer_size, EEPROM_buffer_size)

fig = plt.figure()
ax = plt.axes(projection ='3d')
ax.plot_wireframe(X, Y, dist_matrix_size.T, color ='blue')
ax.set_title('Distance matrix size');
ax.set_xlabel('RAM buffer size (MB)')
ax.set_ylabel('EEPROM buffer size (MB)')
ax.set_zlabel('Distance matrix size (MB)')
plt.show()