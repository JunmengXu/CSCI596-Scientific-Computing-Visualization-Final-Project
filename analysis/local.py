import matplotlib.pyplot as plt
from PIL import Image
import os

# Image file paths, assuming they are in the current working directory
image_files = [
    'analysis/output/beach.jpg',
    'analysis/output/output_beach.jpg_M4_N4.jpg',
    'analysis/output/output_beach.jpg_M4_N16.jpg',
    'analysis/output/output_beach.jpg_M16_N16.jpg',
    'analysis/output/capybara.jpg',
    'analysis/output/output_capybara.jpg_M4_N4.jpg',
    'analysis/output/output_capybara.jpg_M4_N16.jpg',
    'analysis/output/output_capybara.jpg_M16_N16.jpg',
    'analysis/output/Lenna.jpg',
    'analysis/output/output_Lenna.jpg_M4_N4.jpg',
    'analysis/output/output_Lenna.jpg_M4_N16.jpg',
    'analysis/output/output_Lenna.jpg_M16_N16.jpg'
]

# Parameters for each image
params = [
    {'M': 'Original', 'N': 'Original'},
    {'M': 4, 'N': 4},
    {'M': 4, 'N': 16},
    {'M': 16, 'N': 16},
    {'M': 'Original', 'N': 'Original'},
    {'M': 4, 'N': 4},
    {'M': 4, 'N': 16},
    {'M': 16, 'N': 16},
    {'M': 'Original', 'N': 'Original'},
    {'M': 4, 'N': 4},
    {'M': 4, 'N': 16},
    {'M': 16, 'N': 16}
]

# Setup the matplotlib figure and axes based on the number of images
fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(15, 10))

# Remove the gaps between subplots
plt.subplots_adjust(wspace=0.1, hspace=0)

for ax, img_path, param in zip(axes.flatten(), image_files, params):
    # Open the image file
    img = Image.open(img_path)
    # Display the image
    ax.imshow(img)
    # Set the title to the M and N parameters
    ax.set_title(f"M={param['M']}, N={param['N']}")
    # Turn off the axis
    ax.axis('off')

plt.savefig('Extras/local_result.png')
plt.show()
