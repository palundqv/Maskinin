# https://learnopencv.com/t-sne-for-feature-visualization/
from tqdm import tqdm
import numpy as np
from cv2 import imread, imshow, cvtColor, waitKey


# Compute the coordinates of the image on the plot
def compute_plot_coordinates(image, x, y, image_centers_area_size, offset):
    image_height, image_width, _ = image.shape
    # compute the image center coordinates on the plot
    center_x = int(image_centers_area_size * x) + offset
    # in matplotlib, the y axis is directed upward
    # to have the same here, we need to mirror the y coordinate
    center_y = int(image_centers_area_size * (1 - y)) + offset
    # knowing the image center,
    # compute the coordinates of the top left and bottom right corner
    tl_x = center_x - int(image_width / 2)
    tl_y = center_y - int(image_height / 2)
    br_x = tl_x + image_width
    br_y = tl_y + image_height
    return tl_x, tl_y, br_x, br_y

# we'll put the image centers in the central area of the plot
# and use offsets to make sure the images fit the plot
# init the plot as white canvas
tsne_plot = 255 * np.ones((plot_size, plot_size, 3), np.uint8)

# now we'll put a small copy of every image to its corresponding T-SNE coordinate
for image_path, label, x, y in tqdm(zip(images, labels, tx, ty),desc='Building the T-SNE plot',total=len(images)):
    image = imread(image_path)
    # scale the image to put it to the plot
    image = scale_image(image, max_image_size)
    # draw a rectangle with a color corresponding to the image class
    image = draw_rectangle_by_class(image, label)
    # compute the coordinates of the image on the scaled plot visualization
    tl_x, tl_y, br_x, br_y = compute_plot_coordinates(image, x, y, image_centers_area_size, offset)
    # put the image to its t-SNE coordinates using numpy sub-array indices
    tsne_plot[tl_y:br_y, tl_x:br_x, :] = image

imshow('t-SNE', tsne_plot)
waitKey()
