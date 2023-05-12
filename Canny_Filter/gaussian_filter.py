import numpy as np
from scipy.signal import convolve2d

def generate_gaussian_kernel(size:tuple[int], sigma:float) -> np.ndarray:
    """
    Parameters:

        * size: Size of the gaussian matrix (must be odd). It is the amplitude of near pixel that will be affected.
        * sigma: Standard deviation that will be targeted. The more it is large and the more the filter will be wide.
    """
    if size[0] % 2 == 0 or size[1] % 2 == 0:
        raise ValueError("The size of the matrix must be odd.")
    center = (size[0] // 2, size[1] // 2)
    x, y = np.indices(size)
    g = np.exp(-((x - center[0])**2 + (y - center[1])**2) / (2 * sigma**2))
    return g / g.sum()

def apply_gaussian_filter(image, size:tuple[int], sigma:float):
    kernel = generate_gaussian_kernel(size, sigma)
    filtered_image = convolve2d(image, kernel, mode='same')
    return filtered_image