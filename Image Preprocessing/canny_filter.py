import numpy as np
from math import pi
from scipy.signal import convolve2d
from gaussian_filter import apply_gaussian_filter

def non_maximum_suppression(magnitude, direction):
    for y in range(1, len(magnitude)-1):
        for x in range(1,len(magnitude[0])-1):

            rounded_direction = 45 * np.round(direction[y][x] / 45.0)

            if rounded_direction == 0:
                if magnitude[y][x-1] >= magnitude[y][x] or  magnitude[y][x+1] >= magnitude[y][x]:
                    magnitude[y][x] = 0
            elif rounded_direction == 45:
                if magnitude[y+1][x+1] >= magnitude[y][x] or  magnitude[y+1][x-1] >= magnitude[y][x]:
                    magnitude[y][x] = 0
            elif rounded_direction == 90:
                if magnitude[y+1][x] >= magnitude[y][x] or  magnitude[y-1][x] >=  magnitude[y][x]:
                    magnitude[y][x] = 0
            elif rounded_direction == 180:
                if magnitude[y+1][x+1] >= magnitude[y][x] or  magnitude[y-1][x-1] >=  magnitude[y][x]:
                    magnitude[y][x] = 0
            return magnitude

def hysteresis_thresholding(suppressed, low_threshold, high_threshold):
    high_mask = suppressed >= high_threshold
    low_mask = (suppressed >= low_threshold) & (suppressed < high_threshold)
    
    binary_image = np.zeros_like(suppressed)
    binary_image[high_mask] = 255
    
    y, x = np.where(low_mask)

    for i in range(len(y)):
        if (y[i] > 0 and binary_image[y[i]-1, x[i]] == 255) \
        or (y[i] < suppressed.shape[0]-1 and binary_image[y[i]+1, x[i]] == 255) \
        or (x[i] > 0 and binary_image[y[i], x[i]-1] == 255) \
        or (x[i] < suppressed.shape[1]-1 and binary_image[y[i], x[i]+1] == 255) \
        or (y[i] > 0 and x[i] > 0 and binary_image[y[i]-1, x[i]-1] == 255) \
        or (y[i] > 0 and x[i] < suppressed.shape[1]-1 and binary_image[y[i]-1, x[i]+1] == 255) \
        or (y[i] < suppressed.shape[0]-1 and x[i] > 0 and binary_image[y[i]+1, x[i]-1] == 255) \
        or (y[i] < suppressed.shape[0]-1 and x[i] < suppressed.shape[1]-1 and binary_image[y[i]+1, x[i]+1] == 255):
            binary_image[y[i], x[i]] = 255
    return binary_image

def canny_filter(image,low_threshold,high_threshold, size, sigma):
    image = apply_gaussian_filter(image, size, sigma)
    # Sobel Filter
    grad_x = convolve2d(image,np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]), mode='same')
    grad_y = convolve2d(image,np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]), mode='same')
    
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    direction = np.arctan2(grad_y, grad_x)

    magnitude = non_maximum_suppression(magnitude, direction)
    edges = hysteresis_thresholding(magnitude, low_threshold, high_threshold)
    return edges