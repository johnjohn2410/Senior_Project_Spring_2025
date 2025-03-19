import cv2
import numpy as np
import argparse
from scipy.ndimage import label, find_objects, binary_fill_holes

def remove_noise_and_extract_largest(mask, threshold=0.5):
    """
    Removes noise from the mask and extracts the largest connected component.
    Args:
        mask (numpy.ndarray): The predicted mask.
        threshold (float): Threshold to binarize the mask.
    Returns:
        numpy.ndarray: Cleaned mask with only the largest connected component.
    """
    # Binarize the mask
    binary_mask = mask > threshold

    # Label connected components
    labeled_mask, num_features = label(binary_mask)

    # Find the largest connected component
    if num_features > 0:
        largest_component = (labeled_mask == np.argmax(np.bincount(labeled_mask.flat)[1:]) + 1)
        cleaned_mask = binary_fill_holes(largest_component)
    else:
        cleaned_mask = np.zeros_like(mask)

    return cleaned_mask
