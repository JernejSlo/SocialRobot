import cv2
import easyocr
import numpy as np
import re


class TextRecognitionUtils:
    def __init__(self, lang='en'):
        # Load EasyOCR model
        self.text_model = easyocr.Reader([lang], gpu=False)
        self.commands = []  # List of (text, bbox, parsed_command)

    def find_texts(self, image):
        """Detect text and bounding boxes in the image using EasyOCR."""
        self.commands.clear()

        results = self.text_model.readtext(image)

        for bbox, text, confidence in results:
            text = text.strip()
            parsed = self.parse_command(text)
            if parsed:
                # Flatten bbox to (x1, y1, x2, y2) for simplicity
                x_coords = [p[0] for p in bbox]
                y_coords = [p[1] for p in bbox]
                x1, y1 = int(min(x_coords)), int(min(y_coords))
                x2, y2 = int(max(x_coords)), int(max(y_coords))
                flat_bbox = (x1, y1, x2, y2)
                self.commands.append((text, flat_bbox, parsed))

        return self.commands

    def parse_command(self, text):
        """Basic parser for commands like '2 lemons' or 'Place 3 apples'."""
        match = re.search(r'(\d+)\s*(\w+)', text)
        if match:
            count = int(match.group(1))
            fruit = match.group(2).lower()
            return {"count": count, "fruit": fruit}
        return None

    def is_inside_command_area(self, bbox, margin=100):
        """Check if a given bbox (x1, y1, x2, y2) lies inside any command's reserved area."""
        x1, y1, x2, y2 = bbox
        for cmd_text, cmd_bbox, parsed in self.commands:
            cx1, cy1, cx2, cy2 = cmd_bbox
            # Expand bbox by margin
            cx1 -= margin
            cy1 -= margin
            cx2 += margin
            cy2 += margin
            if x1 >= cx1 and y1 >= cy1 and x2 <= cx2 and y2 <= cy2:
                return True, parsed
        return False, None


if __name__ == "__main__":
    image = cv2.imread("CommandImage.jpg")
    ci = TextRecognitionUtils()
    commands = ci.find_texts(image)
    print(commands)

    for cmd_text, bbox, parsed in commands:
        print(f"Found command: '{cmd_text}' -> {parsed} at {bbox}")

    # Example bbox to test if inside any command zone:
    test_bbox = (150, 120, 170, 140)
    inside, info = ci.is_inside_command_area(test_bbox)
    if inside:
        print(f"Bbox {test_bbox} is inside command zone for: {info}")
