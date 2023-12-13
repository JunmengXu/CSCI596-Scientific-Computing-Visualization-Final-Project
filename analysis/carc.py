import matplotlib.pyplot as plt
from PIL import Image

# Assuming the URLs are the paths to the images
image_urls = [
    'analysis/output/beach.jpg',  # Original image
    'analysis/output/beach_4_16_1_2.jpg',
    'analysis/output/beach_4_16_1_4.jpg',
    'analysis/output/beach_4_16_2_2.jpg',
    'analysis/output/beach_4_16_2_4.jpg',
    'analysis/output/beach_4_16_4_2.jpg',
    'analysis/output/beach_4_16_4_4.jpg'
]

# Titles for the table, extracted from the file names
image_titles = [
    'Original',
    '1 Node, 2 Tasks',
    '1 Node, 4 Tasks',
    '2 Nodes, 2 Tasks',
    '2 Nodes, 4 Tasks',
    '4 Nodes, 2 Tasks',
    '4 Nodes, 4 Tasks'
]

# Load images and store in a list
images = [Image.open(url) for url in image_urls]

# Create figure and axes
fig, axs = plt.subplots(1, len(images), figsize=(20, 10))

# Set each title and turn off axis
for ax, img, title in zip(axs, images, image_titles):
    ax.set_title(title)
    ax.imshow(img)
    ax.axis('off')

# Show the plot
plt.tight_layout()
plt.savefig('Extras/carc_result.png')
plt.show()
