

class Result:
    def __init__(self):
        """
        Constructor for result class. Makes lists for all variables I want to fill
        """
        # Euler angles
        self.pitch = []
        self.roll = []
        self.yaw = []

        # Position in 3d
        self.pos_x = []
        self.pos_y = []
        self.pos_z = []

        # Vertical acceleration
        self.acc_x = []
