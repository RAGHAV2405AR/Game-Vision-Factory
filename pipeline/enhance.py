import cv2
import numpy as np


def enhance_frame(img):
    """
    Takes a raw game frame and improves it before YOLO tries to detect objects.

    Why do we need this?
    Game footage is often dark, washed out, or blurry.
    YOLO detects objects by looking for edges and contrast.
    If the image is dark or flat, YOLO misses objects or detects them with low confidence.
    This function fixes that by:
      1. Boosting contrast in dark areas (CLAHE)
      2. Sharpening blurry edges

    Returns the improved image (same size, still BGR format for OpenCV).
    """
    # Convert from BGR (OpenCV default) to LAB
    lab_image = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    # Split into the 3 separate channels
    l_channel, a_channel, b_channel = cv2.split(lab_image)

    # Create the CLAHE tool
    # clipLimit=2.0  - controls how much contrast boost is allowed
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    # Apply CLAHE only to the L (lightness) channel
    l_channel = clahe.apply(l_channel)

    #merge
    lab_image = cv2.merge((l_channel, a_channel, b_channel))

    # Convert back to BGR so the rest of the pipeline can use it
    img = cv2.cvtColor(lab_image, cv2.COLOR_LAB2BGR)


    sharpen_kernel = numpy_sharpen_kernel()
    img = cv2.filter2D(img, -1, sharpen_kernel)
    # -1 means the output image keeps the same bit depth as the input

    return img


def numpy_sharpen_kernel():
    """
    Builds and returns the sharpening kernel as a numpy array.
    Kept separate so it is easy to understand and swap out if needed.
    """
    import numpy as np

    kernel = np.array([  [ 0, -1,  0], [-1,  5, -1],[ 0, -1,  0] ])
    return kernel
