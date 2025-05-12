import cv2
import numpy as np
import matplotlib.pyplot as plt

class RotationUtils():

    def __init__(self):
        pass

    def rotate_point(self, center, point, angle):
        angle = np.deg2rad(angle)

        rotation_matrix = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)]
        ])

        point = np.array(point) - np.array(center)
        rotated_point = np.dot(rotation_matrix, point) + np.array(center)

        return tuple(rotated_point.astype(int))

    def gripper_rotation(self,image, rotation_angle, fruit):

        #########################        DODANO        #################################
        image, edges = self.select_fruit(image, fruit)

        if image is None:
            print("Error: Image not found.")
            exit()

        # ➕ Connect broken edges using dilation
        kernel = np.ones((7, 7), np.uint8)  # Try (5, 5) if edges are really broken
        edges = cv2.dilate(edges, kernel, iterations=1)
        ################################################################################

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            print("No objects detected.")
            exit()

        # Keep only the largest contour
        largest_contour = max(contours, key=cv2.contourArea)

        # Draw only the outermost contour
        contour_image = np.zeros_like(image)
        cv2.drawContours(contour_image, [largest_contour], -1, color=255, thickness=1)

        edges = contour_image  # override edge image with filtered version

        if not contours:
            print("No objects detected.")
            exit()

        largest_contour = max(contours, key=cv2.contourArea)

        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

        output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        height, width = output_image.shape[:2]
        cv2.rectangle(output_image, (0, 0), (width - 1, height - 1), (0, 255, 0), 1)

        cv2.circle(output_image, (cx, cy), 1, (255, 0, 0), -1)

        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        cv2.circle(edges_colored, (cx, cy), 1, (0, 0, 255), -1)

        line_image = np.zeros_like(edges)
        line_image_colored = cv2.cvtColor(line_image, cv2.COLOR_GRAY2BGR)

        line_length = 500
        half_length = line_length // 2
        top_point = (cx, cy - half_length)
        bottom_point = (cx, cy + half_length)

        rotated_top = self.rotate_point((cx, cy), top_point, rotation_angle)
        rotated_bottom = self.rotate_point((cx, cy), bottom_point, rotation_angle)

        cv2.line(line_image_colored, rotated_top, rotated_bottom, (255, 255, 255), 1)

        return edges_colored, line_image_colored

    def find_matching_pixels(self, edges_with_centroid, rotated_line_image):
        condition1 = np.all(edges_with_centroid == [255, 255, 255], axis=-1)
        condition2 = np.all(rotated_line_image == [255, 255, 255], axis=-1)

        matching_condition = condition1 & condition2
        matching_indices = np.where(matching_condition)

        return matching_indices

    def calculate_distances(self, matching_indices, min_distance_threshold):
        y_coords, x_coords = matching_indices

        if len(x_coords) < 2:
            return []

        distances = []
        for i in range(len(x_coords)):
            for j in range(i + 1, len(x_coords)):
                x1, y1 = x_coords[i], y_coords[i]
                x2, y2 = x_coords[j], y_coords[j]
                distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

                if distance >= min_distance_threshold:
                    distances.append(distance)

        return distances

    def select_fruit(self,image, fruit):
        if fruit == "lemon":
            blue_channel = image[:, :, 0]
            _, binary_image = cv2.threshold(blue_channel, 40, 255, cv2.THRESH_BINARY)
            edges = cv2.Canny(binary_image, 150, 150)
            return blue_channel, edges
        elif fruit == "mandarin":
            blue_channel = image[:, :, 0]
            _, binary_image = cv2.threshold(blue_channel, 50, 255, cv2.THRESH_BINARY)
            edges = cv2.Canny(binary_image, 150, 150)
            return blue_channel, edges
        elif fruit == "cucumber":
            blue_channel = image[:, :, 0]
            _, binary_image = cv2.threshold(blue_channel, 90, 255, cv2.THRESH_BINARY)
            edges = cv2.Canny(binary_image, 150, 150)
            return blue_channel, edges
        elif fruit == "yellow bell pepper":
            blue_channel = image[:, :, 0]
            _, binary_image = cv2.threshold(blue_channel, 50, 255, cv2.THRESH_BINARY)
            edges = cv2.Canny(binary_image, 150, 150)
            return blue_channel, edges
        elif fruit == "red bell pepper":
            blue_channel = image[:, :, 0]
            _, binary_image = cv2.threshold(blue_channel, 50, 255, cv2.THRESH_BINARY)
            edges = cv2.Canny(binary_image, 150, 150)
            return blue_channel, edges
        elif fruit == "strawberry":
            blue_channel = image[:, :, 0]
            _, binary_image = cv2.threshold(blue_channel, 50, 255, cv2.THRESH_BINARY)
            edges = cv2.Canny(binary_image, 150, 150)
            return blue_channel, edges
        elif fruit == "pear":
            blue_channel = image[:, :, 0]
            _, binary_image = cv2.threshold(blue_channel, 50, 255, cv2.THRESH_BINARY)
            edges = cv2.Canny(binary_image, 150, 150)
            return blue_channel, edges
        elif fruit == "peach":
            blue_channel = image[:, :, 0]
            _, binary_image = cv2.threshold(blue_channel, 50, 255, cv2.THRESH_BINARY)
            edges = cv2.Canny(binary_image, 150, 150)
            return blue_channel, edges

    def find_best_rotation(self,image, fruit, max_rotation=180, min_distance_threshold=40):
        edges_with_centroid, _ = self.gripper_rotation(image, 0, fruit)

        best_rotation = 0
        smallest_distance = float('inf')

        for rotation_angle in range(0, max_rotation + 1,5):
            _, rotated_line_image = self.gripper_rotation(image, rotation_angle, fruit)

            matching_indices = self.find_matching_pixels(edges_with_centroid, rotated_line_image)

            distances = self.calculate_distances(matching_indices, min_distance_threshold)

            if distances:
                min_distance = min(distances)
                if min_distance < smallest_distance:
                    smallest_distance = min_distance
                    best_rotation = rotation_angle

            rotation_of_gripper = best_rotation - 90

        return best_rotation, smallest_distance, rotation_of_gripper

if __name__ == "__main__" :

    ru = RotationUtils()
    # Load image
    image = cv2.imread("DetectionImage.jpg")

    # Find the best rotation angle and the corresponding smallest distance
    best_rotation, smallest_distance, rotation_of_gripper = ru.find_best_rotation(image)

    # Print results
    print(f"The best rotation angle of gripper is: {rotation_of_gripper}°")
    print(f"The smallest distance between matching points is: {smallest_distance}")

    # Visualization: Plotting the original edges and the best rotated line
    edges_with_centroid, rotated_line_image = ru.gripper_rotation(image, best_rotation)

    # Overlay the two images
    overlay = cv2.addWeighted(edges_with_centroid, 0.7, rotated_line_image, 0.3, 0)

    # Create a figure with one subplot
    fig, ax = plt.subplots(figsize=(10, 8))

    # Show the overlayed image
    ax.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    ax.set_title(f"Overlay of Edges and Required Rotation of Gripper {rotation_of_gripper}°")
    ax.axis("off")

    # Display the image
    plt.tight_layout()
    plt.show()
