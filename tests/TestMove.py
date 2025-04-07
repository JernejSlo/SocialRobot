from xarm.wrapper import XArmAPI
import time


class SocialRobot:

    def __init__(self, config):
        self.position = self.calibrate_position()
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

    def calibrate_position(self):
        # calculate position

        initial_position = {}

        return initial_position

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

    def return_to_initial_position(self):
        # move to initial position
        init_position = self.arm.get_initial_point()
        print(init_position[1])

    def return_to_home(self):
        # joint move
        code = self.arm.set_servo_angle(angle=[0, -45, -45, 0, 90, -45, 0], wait=True, is_radian=False)

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


init_config = {
    "ip": "192.168.65.203"
}
x_min, y_min, x_max, y_max = [0, 0, 50, 50]

config = {
    "movement": {
        "x": 300,
        "y": 0,
        "z": 200,
        "roll": 180,
        "pitch": 0,
        "yaw": 0,
        "speed": 100,  # Moderate speed
        "wait": True
    },
    "bbox": [x_min, y_min, x_max, y_max]
}


Robot = SocialRobot(init_config)

calibration_position = {
        "x": 450,
        "y": 70,
        "z": 340,
        "roll": 0,
        "pitch": 0,
        "yaw": 0,
        "speed": 50,  # Moderate speed
        "wait": True
    },

calibration_config = {
    "movement": calibration_position
}

def test_move(config):
    Robot.move_robot(config["movement"])
    print(f"Moved robot to {config['movement']}")


def test_calculate_movement(config):
    movement = Robot.calculate_movement(config["bbox"])
    print(f"Calculated movement:")
    print(f"{movement}\n")
    print("From Parameters:\n")
    print(config["bbox"])

#calibration test
test_move(calibration_config)

#test_move(config)

Robot.arm.disconnect()