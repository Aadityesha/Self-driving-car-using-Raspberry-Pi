import RPi.GPIO as GPIO
from time import sleep


class Motor:
    """
    Represents a motor driver for a 2- or 4-wheeled robot using the L298n.
    """

    def __init__(self, EnaA, In1A, In2A, EnaB, In1B, In2B):
        """
        Initializes a new instance of the Motor class.

        :param EnaA: The BCM pin number for motor A's enable pin.
        :param In1A: The BCM pin number for motor A's input 1 pin.
        :param In2A: The BCM pin number for motor A's input 2 pin.
        :param EnaB: The BCM pin number for motor B's enable pin.
        :param In1B: The BCM pin number for motor B's input 1 pin.
        :param In2B: The BCM pin number for motor B's input 2 pin.
        """
        self.EnaA = EnaA
        self.In1A = In1A
        self.In2A = In2A
        self.EnaB = EnaB
        self.In1B = In1B
        self.In2B = In2B
        self.speed = 0

        self.setup_gpio()

    def setup_gpio(self):
        """
        Sets up the GPIO pins for the motor driver.
        """
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        GPIO.setup(self.EnaA, GPIO.OUT)
        GPIO.setup(self.In1A, GPIO.OUT)
        GPIO.setup(self.In2A, GPIO.OUT)
        GPIO.setup(self.EnaB, GPIO.OUT)
        GPIO.setup(self.In1B, GPIO.OUT)
        GPIO.setup(self.In2B, GPIO.OUT)
        self.pwmA = GPIO.PWM(self.EnaA, 100)
        self.pwmB = GPIO.PWM(self.EnaB, 100)
        self.pwmA.start(0)
        self.pwmB.start(0)

    def move(self, speed=0.5, turn=0, delay=0):
        """
        Moves the robot at the given speed and turn rate.

        :param speed: The forward/backward speed of the robot, ranging from -1 to 1.
        :param turn: The turning rate of the robot, ranging from -1 to 1.
        :param delay: The time to wait after moving the robot, in seconds.
        """
        speed *= 100
        turn *= 70
        left_speed = speed - turn
        right_speed = speed + turn

        if left_speed > 100:
            left_speed = 100
        elif left_speed < -100:
            left_speed = -100
        if right_speed > 100:
            right_speed = 100
        elif right_speed < -100:
            right_speed = -100

        self.pwmA.ChangeDutyCycle(abs(left_speed))
        self.pwmB.ChangeDutyCycle(abs(right_speed))
        GPIO.output(self.In1A, GPIO.HIGH if left
