import os
import sys
import traceback
import cv2
import keyboard
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO
from xarm.wrapper import XArmAPI
import time
import atexit
from skimage.morphology import medial_axis
import math

from Utils.RotationUtils import RotationUtils
from Utils.VoiceRecognitionUtils import VoiceRecognitionUtils
from Utils.LLMUtils import LLMUtils
from Utils.TextRecognitionUtils import TextRecognitionUtils
from tests.tests import *

class SocialRobot(VoiceRecognitionUtils, LLMUtils, RotationUtils,TextRecognitionUtils):

    def __init__(self, config, **kwargs):
        atexit.register(self.on_exit)

        VoiceRecognitionUtils.__init__(self)
        LLMUtils.__init__(self)
        RotationUtils.__init__(self)
        TextRecognitionUtils.__init__(self)

        # detection values
        self.objects_that_can_be_detected = []
        self.shutdown = False


        self.load_model(config["model_path"])

        # movement values
        self.camera_index = 0
        self.save_directory = "./"

        self.home_position = [-11.5, -55, -33, 0, 88, 169, 0]
        self.base_tip_position = [12.9,10.5,-35.1,-2,25.2,15.4,0]

        self.camera_size = [640,480]

        self.base_tip_xyz_dff_corner = [379.3-170.9,84.2+97.1,0-282.4]
        cone_size = 15
        gripper_offset = 15
        self.base_tip_xyz_dff_center = [(407.4-330),(82.2-87.6)+gripper_offset,(9.6-295.4)-cone_size]
        self.offset_x, self.offset_y = 1.5,0
        self.tip_camera_distance = []
        self.camera_top_left_adjust = []
        self.camera_center_adjust = []
        skip_connection = kwargs.get("skip_connection",False)
        self.skip_connection = skip_connection
        # set from prior calibration
        self.mm_per_pixel = 0.6724

        if not skip_connection:
            print("connected")
            self.arm = XArmAPI(config["ip"])
            time.sleep(0.5)
            # Clean error and warn
            if self.arm.warn_code != 0:
                self.arm.clean_warn()
            if self.arm.error_code != 0:
                self.arm.clean_error()

            # Enable the robot
            self.arm.motion_enable(enable=True)
            # Set mode
            # P0: position control mode
            #     1: servo motion mode
            #     2: joint teaching mode
            #     3: cartesian teaching mode (invalid)
            #     4: joint velocity control mode
            #     5: cartesian velocity control mode
            #     6: joint online trajectory planning mode
            #     7: cartesian online trajectory planning mode
            self.arm.set_mode(0)

            # Set state
            #     0: sport state
            #     3: pause state
            #     4: stop state
            self.arm.set_state(0)

            self.return_to_home()

    def load_model(self,path):
        self.model = YOLO(path)  # Ensure your trained model exists
        self.objects_that_can_be_detected = list(self.model.names.values())

    def find_camera_indices(self):
        for i in range(10):  # Check indices 0-9
            try:
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    print(f"Camera available at index {i}")
                    cap.release()
                    break
            except:
                print(f"Error opening camera {i}")

    def take_picture(self, **kwargs):
        """
        Captures an image from the camera, saves it as 'CalibrationImage.jpg',
        and returns the path to the saved image.

        Returns:
        - str: Path to the saved image file.
        """


        find_indices = kwargs.get("find_indices",False)

        if find_indices:
            self.find_camera_indices()
        if sys.platform.startswith("win"):
            cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        elif sys.platform.startswith("linux"):
            cap = cv2.VideoCapture(self.camera_index)
        else:
            cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        #cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)  # Set width
        #ap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)  # Set height
        if not cap.isOpened():
            print("Error: Could not open camera.")
            return None

        ret, frame = cap.read()
        cap.release()

        if not ret:
            print("Error: Failed to capture image.")
            return None

        img_name = kwargs.get("img_name","CalibrationImage.jpg")
        # Define the image path
        image_path = os.path.join(self.save_directory, img_name)
        self.calib_path = image_path
        # Save the image
        cv2.imwrite(image_path, frame)
        print(f"Calibration image saved: {image_path}")

        return image_path

    def calculate_mm_per_pixel_from_corners(self, board_corners, square_size_mm=14, squares_x=21, squares_y=15):
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

        return mm_per_pixel, mm_per_pixel_x, mm_per_pixel_y

    def calibrate_position(self, **kwargs):
        skip_pic = kwargs.get("skip_taking_picture", False)
        if not skip_pic:
            path = self.take_picture()
        else:
            path = kwargs.get("calib_path","./CalibrationImage.jpg")
            self.calib_path = path

        frame = cv2.imread(path)
        plt.imshow(frame)
        plt.show()
        if frame is None:
            print("❌ Error: Could not load image. Check the file path.")
        else:

            # Convert to grayscale for better corner detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            plt.imshow(gray)
            plt.show()

            # Define chessboard size (number of inner corners, NOT squares)
            chessboard_size = (16, 22)  # Adjust based on your board

            # Find chessboard corners
            found, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

            if found:
                # Extract relevant corner points (Top-left, Top-right, Bottom-left, Bottom-right)
                bottom_left = corners[0][0]  # First point
                bottom_right = corners[chessboard_size[0] - 1][0]  # Last point in the first row
                top_left = corners[-chessboard_size[0]][0]  # First point in the last row
                top_right = corners[-1][0]  # Last point in the last row

                adj = [1000000,100000]

                for p in [bottom_left,top_left,top_right,bottom_right]:
                    if sum(p) < sum(adj):
                        adj = p

                self.camera_top_left_adjust = adj
                print(f"This is the top left: {adj}")

                self.camera_center_adjust = [(adj[0]-(self.camera_size[0]/2)),(adj[1]-(self.camera_size[1]/2))]
                print(f"This is the center: {self.camera_center_adjust}")

                # Store and print only these four points
                board_corners = {
                    "Top Left": top_left,
                    "Top Right": top_right,
                    "Bottom Left": bottom_left,
                    "Bottom Right": bottom_right,
                }
                self.board_corners = board_corners
                print("✅ Chessboard detected!")
                for label, point in board_corners.items():
                    print(f"{label}: {point}")
                self.mm_per_pixel,self.mm_per_pixel_x,self.mm_per_pixel_y = self.calculate_mm_per_pixel_from_corners(board_corners)


                t = f"1 pixel = {self.mm_per_pixel:.4f} mm"
                print(t)

                # Move Center and then tip logic
                self.align_camera(self.camera_center_adjust)
                self.move_tip_to_base(self.base_tip_xyz_dff_center)

                # Move corner and then tip logic
                #self.align_camera(self.camera_top_left_adjust)
                #self.move_tip_to_base(self.base_tip_xyz_dff_corner)
            else:
                print("❌ No chessboard found.")

    def move_tip_to_base(self, base_tip_xyz_dff):
        print("Moving tip to base.")
        initial_position = self.arm.get_position()
        print(f"This is the current position: {initial_position[1]}")
        x,y,z = base_tip_xyz_dff

        print(f"This is the base to camera diff: {base_tip_xyz_dff}")

        x_, y_, z_, roll, pitch, yaw = initial_position[1]


        movement = {
            "x": x_+x,
            "y": y_+y,
            "z": z_+z,
            "roll": roll,
            "pitch": pitch,
            "yaw": yaw,
            "speed": 50,  # Moderate speed
            "wait": True
        }

        self.tip_base_position = movement

        self.move_robot(movement)

    def align_camera(self,adjust):
        initial_position = self.arm.get_position()
        print(f"This is the current position: {initial_position[1]}")

        x,y,z,roll,pitch,yaw = initial_position[1]

        x_adj = adjust[1]*self.mm_per_pixel
        y_adj = adjust[0]*self.mm_per_pixel

        movement = {
            "x": x-x_adj,
            "y": y-y_adj,
            "z": z,
            "roll": roll,
            "pitch": pitch,
            "yaw": yaw,
            "speed": 100,  # Moderate speed
            "wait": True
        }

        self.corner_left_position = movement

        self.move_robot(movement)

    def move_x_y(self,x,y):
        """
                Move the xArm robot based on the given movement dictionary.

                :param movement: Dictionary with movement parameters
                """

        initial_position = self.arm.get_position()
        print(f"This is the current position: {initial_position[1]}")

        x_, y_, z_, roll, pitch, yaw = initial_position[1]

        self.arm.set_position(
            x=x,
            y=y,
            z=z_,
            roll=roll,
            pitch=pitch,
            yaw=yaw,
            speed=50,
            wait=True
        )

    def move_robot(self, movement: dict):
        """
        Move the xArm robot based on the given movement dictionary.

        :param movement: Dictionary with movement parameters
        """
        self.arm.set_position(
            x=movement["x"],
            y=movement["y"],
            z=movement["z"],
            roll=movement["roll"],
            pitch=movement["pitch"],
            yaw=movement["yaw"],
            speed=movement["speed"],
            wait=movement["wait"]
        )


    def find_and_execute_all_commands_on_board(self, **kwargs):

        plot_detection = kwargs.get("plot_detection", False)
        image_path = kwargs.get("image", None)  # User-provided image path
        image_name = kwargs.get("image_name", "DetectionImage.jpg")

        if image_path is None:
            image_path = self.take_picture(img_name=image_name)

        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ Error: Could not load image from {image_path}")
            return

        bboxes, labels = self.detect_all_objects(image=image_path, plot_detection=plot_detection)

        self.find_texts(image)
        print(self.commands)
        assignments = self.assign_detections_to_commands(bboxes,labels)
        if plot_detection:
            self.visualize_commands(image,bboxes,labels)
            self.visualize_assignments_with_arrows(image,assignments)


    def detect_all_objects(self, **kwargs):
        """
        Detects all objects in a camera frame or a provided image using YOLO and returns bounding boxes and labels.

        Args:
            kwargs:
                - plot_detection (bool): If True, shows a matplotlib plot of the detections.
                - confidence_threshold (float): Minimum confidence to accept a detection.
                - image_name (str): Filename to save if capturing from camera.
                - image (str): Optional image path to load instead of capturing.

        Returns:
            Tuple:
                - bboxes (List[List[int]]): Bounding boxes [x1, y1, x2, y2].
                - labels (List[str]): Detected class labels.
        """
        bboxes = []
        labels = []

        confidence_threshold = kwargs.get("confidence_threshold", 0.4)
        plot_detection = kwargs.get("plot_detection", False)
        image_path = kwargs.get("image", None)  # User-provided image path
        image_name = kwargs.get("image_name", "DetectionImage.jpg")

        if image_path is None:
            image_path = self.take_picture(img_name=image_name)

        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ Error: Could not load image from {image_path}")
            return [], []

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.model(image_path)
        colors = {}

        for result in results:
            names = [result.names[cls.item()] for cls in result.boxes.cls.int()]
            for i, (xyxy, conf, cls) in enumerate(zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls.int())):
                if conf < confidence_threshold:
                    continue

                x1, y1, x2, y2 = map(int, xyxy.tolist())
                bboxes.append([x1, y1, x2, y2])
                labels.append(names[i])

                if plot_detection:
                    label_text = f"{names[i]} {conf:.2f}"
                    if names[i] not in colors:
                        colors[names[i]] = tuple(np.random.randint(0, 255, 3).tolist())
                    color = colors[names[i]]
                    cv2.rectangle(image_rgb, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(image_rgb, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        if plot_detection:
            plt.figure(figsize=(12, 8))
            plt.imshow(image_rgb)
            plt.axis("off")
            plt.title("All Detected Objects")
            plt.show()

        return bboxes, labels

    def detect_selected(self,objects_to_find,**kwargs):
        bboxes = []
        labels = []
        image_path = self.take_picture(img_name="DetectionImage.jpg")

        plot_detection = kwargs.get("plot_detection", False)
        skip_search = kwargs.get("skip_search", False)

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for Matplotlib

        # Run YOLOv8 inference
        results = self.model(image_path)

        # Define colors for each class
        colors = {}
        print(f"Searching for {objects_to_find}")
        #print("Result:")
        #print(results)
        # Draw the detected bounding boxes
        for result in results:
            names = [result.names[cls.item()] for cls in result.boxes.cls.int()]

            for i, (xyxy, conf, cls) in enumerate(zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls.int())):

                if names[i].lower() in objects_to_find and conf > 0.4:
                    print("################################################")
                    print("Found matching object")
                    print("################################################")
                    x1, y1, x2, y2 = map(int, xyxy.tolist())  # Convert to integers

                    bboxes.append([x1, y1, x2, y2])
                    labels.append([names[i], conf])
                    if plot_detection:
                        label = f"{names[i]} {conf:.2f}"  # Class name and confidence

                        # Assign a unique color for each class
                        if names[i] not in colors:
                            colors[names[i]] = tuple(np.random.randint(0, 255, 3).tolist())

                        # Draw bounding box
                        cv2.rectangle(image, (x1, y1), (x2, y2), colors[names[i]], 2)

                        # Draw label background
                        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                        text_x, text_y = x1, y1 - 5 if y1 - 5 > 10 else y1 + 15
                        cv2.rectangle(image, (text_x, text_y - text_size[1] - 3), (text_x + text_size[0], text_y + 3),
                                      colors[names[i]], -1)
                        height, width = image.shape[:2]

                        cv2.line(image, (width // 2, 0), (width // 2, height), (0, 255, 0), 2)
                        cv2.line(image, (0, height // 2), (width, height // 2), (0, 255, 0), 2)
                        # Draw label text
                        cv2.putText(image, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        found_with_search = False
        if len(bboxes) == 0 and not skip_search:
            self.check_board_for_object(objects_to_find)
            found_with_search = True

        if plot_detection:
            # Display the image with annotations
            plt.figure(figsize=(10, 8))
            plt.imshow(image)
            plt.axis("off")
            plt.show()
        print("result of search is: ")
        print(bboxes,labels)
        return bboxes, labels, found_with_search

    def save_image_bbox(self, bbox, img, save_path):
        """
        Save a cropped region of self.image defined by bbox to the given path.

        :param bbox: Tuple (x1, y1, x2, y2) specifying the bounding box
        :param path: File path to save the cropped image (e.g. 'cropped.jpg')
        """
        print(bbox)
        x_min, y_min, x_max, y_max = bbox

        # Make sure image exists
        if img is None:
            print("❌ self.image is not defined.")
            return

        # Crop the image
        cropped = img[y_min:y_max, x_min:x_max]

        # Save to path
        success = cv2.imwrite(save_path, cropped)
        if success:
            print(f"✅ Cropped image saved to {save_path}")
        else:
            print(f"❌ Failed to save image to {save_path}")

    def align_with_center_of_bbox(self, bbox, object, rotate=False, objects_to_find=[]):
        """
        Aligns camera with the center of the given bounding box.

        Args:
            bbox (tuple or list): Bounding box defined as (x_min, y_min, x_max, y_max).

        """
        x_min, y_min, x_max, y_max = bbox

        # Calculate the center of the bounding box
        x = (x_min + x_max) / 2
        y = (y_min + y_max) / 2

        self.align_camera([x-self.camera_size[0]/2,y-self.camera_size[1]/2])

        if rotate:
            img = cv2.imread("DetectionImage.jpg")
            self.save_image_bbox(bbox, img, "SampleImagesForBBox/DetectionImageBBox.jpg")
            img = cv2.imread("SampleImagesForBBox/DetectionImageBBox.jpg")
            best_rotation, smallest_distance, rotation_of_gripper = self.find_best_rotation(img,object)
            angle = rotation_of_gripper

            print("Centering before grabbing!")
            x, y = self.calculate_rotation_offset(angle, self.offset_x, self.offset_y)
            print(f"New offset at 90°: x = {x:.2f} cm, y = {y:.2f} cm")

            initial_position = self.arm.get_position()
            x_, y_, z, roll, pitch, yaw = initial_position[1]

            x, y = x*10 + x_, y*10 + y_
            self.move_x_y(x, y)

            print(f"Rotating to angle {angle}")
            self.rotate_camera(angle)

    def calculate_rotation_offset(self,angle_deg, offset_x, offset_y):
            """
            Calculates the new x, y position of an offset point after rotating by angle_deg.

            :param angle_deg: The rotation angle in degrees (positive = counterclockwise)
            :param offset_x: The original x offset from the center of rotation
            :param offset_y: The original y offset from the center of rotation
            :return: (new_x, new_y) position of the offset after rotation
            """
            angle_rad = math.radians(angle_deg)

            new_x = offset_x * math.cos(angle_rad) - offset_y * math.sin(angle_rad)
            new_y = offset_x * math.sin(angle_rad) + offset_y * math.cos(angle_rad)

            return new_x, new_y

    def rotate_camera(self,angle):

        initial_position = self.arm.get_position()
        print(f"This is the current position: {initial_position[1]}")

        x, y, z, roll, pitch, yaw = initial_position[1]

        movement = {
            "x": x,
            "y": y,
            "z": z,
            "roll": roll,
            "pitch": pitch,
            "yaw": 180-angle,
            "speed": 100,  # Moderate speed
            "wait": True
        }

        self.move_robot(movement)

    def detect_objects_and_move_to_first(self,objects_to_find,**kwargs):
        try:
            found_with_search = False

            plot_detection = kwargs.get("plot_detection",False)
            skip_search = kwargs.get("skip_search", False)

            bboxes, labels, found_with_search = self.detect_selected(objects_to_find,plot_detection=plot_detection,skip_search=skip_search)

            if found_with_search:
                print("Found the object with search 1.")
                return

            bbox = np.asarray(bboxes[0])
            print(labels)
            label = labels[0][0].lower()
            # Move camera center above the lemon and move the tip down to it
            self.align_with_center_of_bbox(bbox,label)

            # test to see
            bboxes, labels, found_with_search = self.detect_selected(objects_to_find,plot_detection=plot_detection,skip_search=skip_search)
            bbox = np.asarray(bboxes[0])
            self.align_with_center_of_bbox(bbox,label)

            if found_with_search:
                print("Found the object with search 2.")
                return


            bboxes, labels, found_with_search = self.detect_selected(objects_to_find,plot_detection=plot_detection,skip_search=skip_search)
            bbox = np.asarray(bboxes[0])
            self.align_with_center_of_bbox(bbox,label, rotate=True,objects_to_find=objects_to_find)

            if found_with_search:
                print("Found the object with search 3.")
                return

            self.open_grabber()
            self.move_tip_to_base(self.base_tip_xyz_dff_center)
            self.close_grabber()
            time.sleep(1)
        except Exception as e:
            if not found_with_search:
                print("Unable to find the selected object. Moving back to home position.")
                print(f"Exception occurred: {e}")
                traceback.print_exc()
            else:
                print()

        # move the robot arm back up
        self.return_to_home()

    def open_grabber(self):
        self.arm.set_tgpio_digital(0, 1, delay_sec=0)

    def close_grabber(self):
        self.arm.set_tgpio_digital(0, 0, delay_sec=0)

    def correct_perspective_distortion(self):
        """
        Corrects the perspective distortion without cropping or transforming fully to the board corners.

        Returns:
        - corrected_image: Perspective-corrected image, keeping full frame.
        """

        board_corners = self.board_corners

        if not all(key in board_corners for key in ["Bottom Left", "Bottom Right", "Top Left", "Top Right"]):
            raise ValueError("Missing required board corners")

        # Load the original image
        frame = cv2.imread(self.calib_path)
        height, width = frame.shape[:2]

        # Get the detected chessboard corners
        pts_src = np.array([
            board_corners["Top Left"],
            board_corners["Top Right"],
            board_corners["Bottom Left"],
            board_corners["Bottom Right"]
        ], dtype=np.float32)

        # Define the destination points to be **only slightly adjusted**
        # Instead of warping to a small cropped rectangle, keep the board size close to its original shape
        pts_dst = np.array([
            [pts_src[0][0] + 20, pts_src[0][1] + 20],  # Shifted slightly
            [pts_src[1][0] - 20, pts_src[1][1] + 20],
            [pts_src[2][0] + 20, pts_src[2][1] - 20],
            [pts_src[3][0] - 20, pts_src[3][1] - 20]
        ], dtype=np.float32)

        # Compute the perspective transformation matrix
        M = cv2.getPerspectiveTransform(pts_src, pts_dst)

        # Apply the transformation to the FULL IMAGE
        corrected_image = cv2.warpPerspective(frame, M, (width, height))

        return corrected_image

    def return_to_initial_position(self):
        # move to initial position
        init_position = self.arm.get_initial_point()
        print(init_position[1])

    def return_to_home(self):
        # joint move
        if not self.skip_connection or self.skip_connection is None:
            code = self.arm.set_servo_angle(angle=self.home_position, speed=100, wait=True, is_radian=False)

    def calculate_movement(self, bbox):
        """
        Calculate movement parameters based on the bounding box.

        :param bbox: A list/tuple of [x_min, y_min, x_max, y_max] (bounding box)
        :return: Dictionary with xArm movement parameters
        """

        # Extract bounding box coordinates
        x_min, y_min, x_max, y_max = bbox

        # Compute center position (example logic)
        c_x = (x_min + x_max) / 2  # Center X
        c_y = (y_min + y_max) / 2  # Center Y

        # Compute actual movement
        x = 0
        y = 0
        z = 0

        # Compute rotation (example: align with bounding box size)
        roll = 0
        pitch = 0
        yaw = 0

        # Define movement parameters
        movement = {
            "x": x,
            "y": y,
            "z": z,
            "roll": roll,
            "pitch": pitch,
            "yaw": yaw,
            "speed": 100,  # Moderate speed
            "wait": True
        }

        return movement

    def check_board_for_object(self, objects_to_find):

        points = [[205,-52],[315,-52],[425,-52]]
        grab_object = False
        for (x,y) in points:
            self.move_x_y(x,y)
            bboxes, labels, found_with_search = self.detect_selected(objects_to_find,skip_search=True)
            if len(labels) != 0:
                print(labels,objects_to_find)
                for label in labels:
                    if label[0].lower() in objects_to_find:
                        grab_object = True
                        break

        if grab_object:
            self.detect_objects_and_move_to_first(objects_to_find,skip_search=True)

    def place_down_in_direction(self,direction):

        initial_position = self.arm.get_position()
        print(f"This is the current position: {initial_position[1]}")

        x, y, z, roll, pitch, yaw = initial_position[1]
        x_, y_ = 0,0
        if direction == "left":
            y_ = 10
        elif direction == "right":
            y_ = -10
        elif direction == "up":
            x_ = 10
        elif direction == "down":
            x_ = -10

        x_,y_ = x+x_,y+y_

        self.move_x_y(x_,y_)
        self.open_grabber()
        self.move_tip_to_base(self.base_tip_xyz_dff_center)
        self.close_grabber()
        self.return_to_home()
    def execute_command(self,command):
        action_ = command.get("action", "")
        object_ = command.get("object", "")
        target_ = command.get("target", "")
        if object_.lower() == "lemon":
            self.detect_objects_and_move_to_first([object_.lower()],plot_detection=True)
            if action_.lower() in ["left","right","up","down"]:
                self.place_down_in_direction(action_)

    def check_for_command(self):
        try:
            joined = ' '.join(self.word_buffer)
            if set(self.word_buffer) & set([word.lower() for word in self.objects_that_can_be_detected]):
                self.detected_command = self.parse_goal(self.extracted)
                self.execute_command(self.detected_command)
                #self.detect_objects_and_move_to_first(plot_detection=True)
        except Exception as e:
            print(f"Exception occurred: {e}")

    def wait_for_command(self):
        print("🎙️ Listening... Press 'q' to quit.")

        try:
            while not keyboard.is_pressed('q'):
                self.extracted = self.listen()

                if self.extracted:
                    print(f"🔑 Keywords: {self.extracted}")
                    self.word_buffer = self.extracted
                    self.silence_loops = 0
                    self.stream.stop_stream()
                    time.sleep(1)
                    self.check_for_command()
                    self.stream.start_stream()
                else:
                    self.silence_loops += 1
                    print(f"🤫 No keywords detected. (Silence loops: {self.silence_loops}/{self.max_silence_loops})")

                if self.silence_loops >= self.max_silence_loops:
                    print("🧹 No activity for a while. Clearing word buffer.")
                    self.word_buffer = []
                    self.silence_loops = 0

                time.sleep(0.1)

        except KeyboardInterrupt:
            print("🛑 Interrupted by user.")

        finally:
            self.cleanup()
    def on_exit(self):
        self.return_to_home()


def test_move(config):
    Robot.move_robot(config["movement"])
    print(f"Moved robot to {config['movement']}")


def test_calculate_movement(config):
    movement = Robot.calculate_movement(config["bbox"])
    print(f"Calculated movement:")
    print(f"{movement}\n")
    print("From Parameters:\n")
    print(config["bbox"])



def execute_written_commands(img):
    Robot.find_and_execute_all_commands_on_board(image=img, plot_detection=True)

init_config = {
    "ip": "192.168.65.203",
    "model_path": "runs/detect/train28/weights/best.pt",
}


Robot = SocialRobot(init_config, skip_connection=True)

#Robot.find_camera_indices(Robot)

#listen_test(Robot)
#test_board_check(Robot)
#test_grab_lemon(Robot)
#test_grab_any(Robot)

execute_written_commands("SR_Datasets/Other/T3.jpg")

#img = cv2.imread("SampleImagesForBBox/DetectionImageBBox.jpg")
#print(Robot.calculate_rotation(img))

#test_calibrate()
#test_warp()
#Robot.arm.disconnect()