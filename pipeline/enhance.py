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

    # -------------------------------------------------------------------------
    # STEP 1 - CONTRAST BOOST USING CLAHE
    # -------------------------------------------------------------------------
    # CLAHE = Contrast Limited Adaptive Histogram Equalization
    # In plain English: it makes dark regions brighter without blowing out
    # the parts that are already bright.
    #
    # Why not just do img * 1.5 (brightness multiply)?
    # That would make EVERYTHING brighter - already bright areas go white.
    # CLAHE is smarter - it only boosts areas that are actually dark.
    #
    # We have to work in LAB color space because CLAHE only works on
    # a single brightness channel, not on color.
    # LAB splits an image into:
    #   L = Lightness (brightness)  <-- we only touch this
    #   A = Green-Red color info
    #   B = Blue-Yellow color info

    # Convert from BGR (OpenCV default) to LAB
    lab_image = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    # Split into the 3 separate channels
    l_channel, a_channel, b_channel = cv2.split(lab_image)

    # Create the CLAHE tool
    # clipLimit=2.0  - controls how much contrast boost is allowed
    #                  higher = more boost but can look unnatural
    # tileGridSize   - divides image into 8x8 tiles and boosts each tile
    #                  separately, so a dark corner gets boosted even if
    #                  the center is already bright
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    # Apply CLAHE only to the L (lightness) channel
    l_channel = clahe.apply(l_channel)

    # Put the channels back together
    lab_image = cv2.merge((l_channel, a_channel, b_channel))

    # Convert back to BGR so the rest of the pipeline can use it
    img = cv2.cvtColor(lab_image, cv2.COLOR_LAB2BGR)

    # -------------------------------------------------------------------------
    # STEP 2 - SHARPENING
    # -------------------------------------------------------------------------
    # A kernel is a small grid of numbers that we slide over the image.
    # Each pixel gets replaced by a weighted average of itself and its neighbors.
    #
    # This specific kernel:
    #   [[ 0, -1,  0],
    #    [-1,  5, -1],
    #    [ 0, -1,  0]]
    #
    # The center value (5) boosts the current pixel.
    # The -1 values subtract the surrounding pixels.
    # Net effect: edges and fine details become sharper and more visible.
    # This helps YOLO see object boundaries more clearly.

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

    kernel = np.array([
        [ 0, -1,  0],
        [-1,  5, -1],
        [ 0, -1,  0]
    ])
    return kernel