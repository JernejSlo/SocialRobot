import cv2
import matplotlib.pyplot as plt
import numpy as np

def warp_chessboard_image(image, board_corners, output_size=(640, 480)):
    """
    Warps the chessboard image to correct perspective using the detected four key corners.

    Parameters:
    - image: The original image containing the chessboard.
    - board_corners: Dictionary with the four main corners of the chessboard.
    - output_size: Tuple (width, height) of the desired output image size.

    Returns:
    - warped_image: The perspective-corrected chessboard image.
    """
    if not all(key in board_corners for key in ["Top Left", "Top Right", "Bottom Left", "Bottom Right"]):
        raise ValueError("Missing required board corners")

    # Convert corner points to a numpy array
    pts_src = np.array([
        board_corners["Top Left"],
        board_corners["Top Right"],
        board_corners["Bottom Left"],
        board_corners["Bottom Right"]
    ], dtype=np.float32)

    # Define the destination points (output image should be a perfect rectangle)
    width, height = output_size
    pts_dst = np.array([
        [0, 0],        # Top Left
        [width - 1, 0],    # Top Right
        [0, height - 1],   # Bottom Left
        [width - 1, height - 1]  # Bottom Right
    ], dtype=np.float32)

    # Compute the perspective transform matrix
    M = cv2.getPerspectiveTransform(pts_src, pts_dst)

    # Apply the warp transformation
    warped_image = cv2.warpPerspective(image, M, (width, height))

    return warped_image

# Function is ready to be used in the pipeline. 🚀♟️ Let me know if you need further adjustments!
img = warp_chessboard_image("./CalibrationImage.jpg")

plt.imshow(img)
plt.show()