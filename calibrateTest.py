"""import cv2
import numpy as np
import time

idx = 0

# Define chessboard size (number of inner corners, NOT squares)
chessboard_size = (16,22)  # Adjust based on your board

# Capture image from camera
cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)  # Adjust if using an external camera

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame")
        break

    # Convert to grayscale for better corner detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Find chessboard corners
    found, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    if found:
        # Extract relevant corner points (Top-left, Top-right, Bottom-left, Bottom-right)
        top_left = corners[0][0]  # First point
        top_right = corners[chessboard_size[0] - 1][0]  # Last point in the first row
        bottom_left = corners[-chessboard_size[0]][0]  # First point in the last row
        bottom_right = corners[-1][0]  # Last point in the last row

        # Store and print only these four points
        board_corners = {
            "Top Left": top_left,
            "Top Right": top_right,
            "Bottom Left": bottom_left,
            "Bottom Right": bottom_right,
        }

        print("✅ Chessboard detected!")
        for label, point in board_corners.items():
            print(f"{label}: {point}")

        # Draw circles on the key corners
        for point in board_corners.values():
            cv2.circle(frame, tuple(point.astype(int)), 5, (0, 0, 255), -1)  # Red dot

    else:
        print("❌ No chessboard found.")

    # Show the detected chessboard with key points highlighted
    cv2.imshow("Chessboard Key Corners", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    time.sleep(1)

# Release resources
cap.release()
cv2.destroyAllWindows()
"""

import numpy as np


def calculate_mm_per_pixel_from_corners(board_corners, square_size_mm=14, squares_x=21, squares_y=15):
    """
    Calculates the mm per pixel ratio based on detected chessboard key corners.

    Parameters:
    - board_corners: Dictionary containing the four main corners of the chessboard.
    - square_size_mm: The real-world size of one chessboard square in mm.
    - squares_x: Number of squares between corners in the long direction.
    - squares_y: Number of squares between corners in the short direction.

    Returns:
    - mm_per_pixel: Average ratio of mm per pixel.
    """
    if not all(key in board_corners for key in ["Top Left", "Top Right", "Bottom Left", "Bottom Right"]):
        raise ValueError("Missing required board corners")

    # Convert to numpy arrays for easy calculations
    top_left = np.array(board_corners["Top Left"])
    top_right = np.array(board_corners["Top Right"])
    bottom_left = np.array(board_corners["Bottom Left"])
    bottom_right = np.array(board_corners["Bottom Right"])

    # Calculate pixel distances
    pixel_width = np.linalg.norm(top_right - top_left)  # Distance across the long side
    pixel_height = np.linalg.norm(bottom_left - top_left)  # Distance across the short side

    # Compute mm per pixel ratios
    mm_per_pixel_x = (squares_x * square_size_mm) / pixel_width
    mm_per_pixel_y = (squares_y * square_size_mm) / pixel_height

    # Compute the average mm per pixel ratio
    mm_per_pixel = (mm_per_pixel_x + mm_per_pixel_y) / 2

    return mm_per_pixel

# This function can now take `board_corners` as input and return the real-world mm per pixel conversion. 🚀

# This function takes the detected chessboard corners and returns the conversion factor.
# To use it, pass the detected corners from the chessboard detection function. 🚀♟️


import cv2
import numpy as np

# Load image from file
image_path = "C:/Users/jerne/OneDrive/Slike/Mapa fotoaparata/calibrateChessBoard.jpg"  # Replace with the actual path to your image

frame = cv2.imread(image_path)

if frame is None:
    print("❌ Error: Could not load image. Check the file path.")
else:
    # Resize image to fit screen (e.g., 800px width while maintaining aspect ratio)
    scale_factor = 800 / frame.shape[1]  # Scale based on width
    new_size = (800, int(frame.shape[0] * scale_factor))
    frame = cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)

    # Convert to grayscale for better corner detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Define chessboard size (number of inner corners, NOT squares)
    chessboard_size = (16, 22)  # Adjust based on your board

    # Find chessboard corners
    found, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    if found:
        # Extract relevant corner points (Top-left, Top-right, Bottom-left, Bottom-right)
        top_left = corners[0][0]  # First point
        top_right = corners[chessboard_size[0] - 1][0]  # Last point in the first row
        bottom_left = corners[-chessboard_size[0]][0]  # First point in the last row
        bottom_right = corners[-1][0]  # Last point in the last row

        # Store and print only these four points
        board_corners = {
            "Top Left": top_left,
            "Top Right": top_right,
            "Bottom Left": bottom_left,
            "Bottom Right": bottom_right,
        }

        print("✅ Chessboard detected!")
        for label, point in board_corners.items():
            print(f"{label}: {point}")
        mm_per_pixel = calculate_mm_per_pixel_from_corners(board_corners)
        t = f"1 pixel = {mm_per_pixel:.4f} mm"

        # Draw circles on the key corners
        for point in board_corners.values():
            cv2.circle(frame, tuple(point.astype(int)), 5, (0, 0, 255), -1)  # Red dot
        cv2.putText(frame, t, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    else:
        print("❌ No chessboard found.")

    # Show the resized image with detected key corners
    cv2.imshow("Chessboard Key Corners", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
