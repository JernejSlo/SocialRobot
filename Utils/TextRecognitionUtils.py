import cv2
from paddleocr import PaddleOCR
import matplotlib
import numpy as np
import re

import matplotlib.pyplot as plt

class TextRecognitionUtils:
    def __init__(self, lang='en'):
        # Load EasyOCR model
        self.text_model = PaddleOCR(use_angle_cls=True, lang=lang)
        self.commands = []  # List of (text, bbox, parsed_command)
        self.margin = 50

    def find_texts(self, image):
        """
        Detect text and bounding boxes in the image using PaddleOCR.

        Assumes that self.text_model is an instance of PaddleOCR.
        """
        self.commands.clear()

        # Convert to file path if needed
        if isinstance(image, str):
            results = self.text_model.ocr(image, cls=True)
        else:
            import cv2
            from tempfile import NamedTemporaryFile

            # Save OpenCV image to a temporary file
            with NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                cv2.imwrite(tmp.name, image)
                results = self.text_model.ocr(tmp.name, cls=True)

        for line in results[0]:
            box_points = line[0]
            text = line[1][0].strip()
            confidence = line[1][1]

            parsed = self.parse_command(text)
            if parsed:
                x_coords = [int(p[0]) for p in box_points]
                y_coords = [int(p[1]) for p in box_points]
                flat_bbox = (min(x_coords), min(y_coords), max(x_coords), max(y_coords))
                self.commands.append((text, flat_bbox, parsed))

        return self.commands

    def visualize_assignments_with_arrows(self, image, assignments):
        """
        Visualize assignments by drawing arrows from each object to its assigned command box.

        Args:
            image: BGR image (as read by OpenCV).
            assignments: List of (bbox, label, target_bbox) as from assign_detections_to_commands.
        """
        vis = image.copy()

        for det_bbox, label, target_bbox in assignments:
            dx1, dy1, dx2, dy2 = det_bbox
            cx1, cy1, cx2, cy2 = target_bbox
            cx1, cy1, cx2, cy2 = cx1-self.margin, cy1-self.margin, cx2+self.margin, cy2+self.margin

            # Centers
            det_center = (int((dx1 + dx2) / 2), int((dy1 + dy2) / 2))
            target_center = (int((cx1 + cx2) / 2), int((cy1 + cy2) / 2))

            # Draw detection box and label
            cv2.rectangle(vis, (dx1, dy1), (dx2, dy2), (0, 0, 255), 2)
            cv2.putText(vis, label, (dx1, dy1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # Draw target command box
            cv2.rectangle(vis, (cx1, cy1), (cx2, cy2), (0, 255, 0), 2)

            # Draw arrow
            cv2.arrowedLine(vis, det_center, target_center, (255, 0, 0), 3, tipLength=0.3)

        # Convert to RGB and show with matplotlib
        vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(12, 8))
        plt.imshow(vis_rgb)
        plt.title("Object → Command Assignment Visualization")
        plt.axis("off")
        plt.show()

    def assign_detections_to_commands(self, bboxes, labels):
        """
        Determines which detected objects need to be moved to which commands on the paper.

        Args:
            bboxes: List of bounding boxes [x1, y1, x2, y2].
            labels: List of corresponding class labels.
            self.margin: Margin around command bounding box to define its "zone".

        Returns:
            A list of tuples: (bbox, label, target_command_bbox)
            Each entry means: this detection should be moved to that command area.
        """
        # Count how many items are already placed per command
        placed_counts = [0 for _ in self.commands]
        to_move = []

        for det_label, det_bbox in zip(labels, bboxes):
            # Check if it touches any command zone
            inside_any = False
            dx1, dy1, dx2, dy2 = det_bbox

            for i, (cmd_text, cmd_bbox, parsed) in enumerate(self.commands):
                if det_label.lower() != parsed["fruit"]:
                    continue

                # Expand command bbox for margin
                x1, y1, x2, y2 = cmd_bbox
                x1 -= self.margin
                y1 -= self.margin
                x2 += self.margin
                y2 += self.margin

                # Check for intersection
                overlap_x = max(0, min(dx2, x2) - max(dx1, x1))
                overlap_y = max(0, min(dy2, y2) - max(dy1, y1))
                if overlap_x > 0 and overlap_y > 0:
                    placed_counts[i] += 1
                    inside_any = True
                    break

            if not inside_any:
                to_move.append((det_label.lower(), det_bbox))

        # Assign objects to command zones that still need items
        assignment = []
        for det_label, det_bbox in to_move:
            best_idx = None
            best_distance = float('inf')
            for i, (cmd_text, cmd_bbox, parsed) in enumerate(self.commands):
                if det_label != parsed["fruit"]:
                    continue
                needed = parsed["count"] - placed_counts[i]
                if needed <= 0:
                    continue

                # Compute distance to command bbox center
                cx1, cy1, cx2, cy2 = cmd_bbox
                cmd_center = ((cx1 + cx2) / 2, (cy1 + cy2) / 2)
                dx1, dy1, dx2, dy2 = det_bbox
                det_center = ((dx1 + dx2) / 2, (dy1 + dy2) / 2)
                dist = np.linalg.norm(np.array(det_center) - np.array(cmd_center))

                if dist < best_distance:
                    best_distance = dist
                    best_idx = i

            if best_idx is not None:
                placed_counts[best_idx] += 1
                assignment.append((det_bbox, det_label, self.commands[best_idx][1]))  # (bbox, label, target_bbox)

        return assignment

    def visualize_commands(self, image, bboxes, labels, show_detections=True):
        """
        Visualize command zones and optionally detections using OpenCV for drawing
        and matplotlib for displaying.

        Args:
            image: The original BGR image (as read by OpenCV).
            bboxes: List of bounding boxes [x1, y1, x2, y2].
            labels: List of corresponding class labels (same length as bboxes).
            show_detections: Whether to draw detection boxes.
        """
        vis = image.copy()

        for cmd_text, cmd_bbox, parsed in self.commands:
            x1, y1, x2, y2 = cmd_bbox
            # Draw command box with margin
            cv2.rectangle(vis, (x1 - self.margin, y1 - self.margin), (x2 + self.margin, y2 + self.margin), (0, 255, 0),
                          2)
            cv2.putText(vis, cmd_text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Count matching detections inside command zone
            fruit_type = parsed["fruit"]
            count = 0
            for det_label, (dx1, dy1, dx2, dy2) in zip(labels, bboxes):
                if det_label.lower() != fruit_type:
                    continue
                # Check intersection
                overlap_x = max(0, min(dx2, x2 + self.margin) - max(dx1, x1 - self.margin))
                overlap_y = max(0, min(dy2, y2 + self.margin) - max(dy1, y1 - self.margin))
                if overlap_x > 0 and overlap_y > 0:
                    count += 1

            count_text = f"{count} {fruit_type}(s) detected"
            cv2.putText(vis, count_text, (x1, y2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

        if show_detections:
            for label, (x1, y1, x2, y2) in zip(labels, bboxes):
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 1)
                cv2.putText(vis, label, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # Convert to RGB and show using matplotlib
        vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(12, 8))
        plt.imshow(vis_rgb)
        plt.axis("off")
        plt.title("Command Visualization")
        plt.show()

    def parse_command(self, text):
        """Basic parser for commands like '2 lemons' or 'Place 3 apples'."""
        match = re.search(r'(\d+)\s*(\w+)', text)
        if match:
            count = int(match.group(1))
            fruit = match.group(2).lower()
            return {"count": count, "fruit": fruit}
        return None

    def is_inside_command_area(self, bbox):
        """Check if a given bbox (x1, y1, x2, y2) lies inside any command's reserved area."""
        x1, y1, x2, y2 = bbox
        for cmd_text, cmd_bbox, parsed in self.commands:
            cx1, cy1, cx2, cy2 = cmd_bbox
            # Expand bbox by margin
            cx1 -= self.margin
            cy1 -= self.margin
            cx2 += self.margin
            cy2 += self.margin
            if x1 >= cx1 and y1 >= cy1 and x2 <= cx2 and y2 <= cy2:
                return True, parsed
        return False, None


if __name__ == "__main__":
    image = cv2.imread("CommandImage.jpg")
    ci = TextRecognitionUtils()
    ci.find_texts(image)

