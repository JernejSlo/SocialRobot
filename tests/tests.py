from matplotlib import pyplot as plt


def test_calibrate(Robot):
    Robot.calibrate_position()

def test_warp(Robot):
    Robot.calibrate_position(skip_taking_picture=False)
    img = Robot.correct_perspective_distortion()
    plt.imshow(img)
    plt.show()

def test_grab_any(Robot):
    #Robot.calibrate_position()
    Robot.detect_objects_and_move_to_first(["lemon","cucumber","strawberry","pear","peach","palm","mandarin","yellow bell pepper","red bell pepper"],plot_detection=True)


def test_grab_lemon(Robot):
    #Robot.calibrate_position()
    Robot.detect_objects_and_move_to_first(["lemon"],plot_detection=False)

def listen_test(Robot):
    Robot.wait_for_command()

def test_board_check(Robot):
    Robot.check_board_for_object(["lemon"])