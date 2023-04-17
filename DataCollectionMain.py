import cv2
import time
import MotorModule as mM
import DataCollectionModule as dcM
import JoyStickModule as jsM
import WebcamModule as wM


MAX_THROTTLE = 0.25
MOTOR_PINS = (2, 3, 4, 17, 22, 27)
RECORDING_DELAY = 0.3


def main():
    motor = mM.Motor(*MOTOR_PINS)
    is_recording = False
    record_start_time = None
    
    while True:
        joystick = jsM.getJS()
        steering = joystick['axis1']
        throttle = joystick['o'] * MAX_THROTTLE

        if joystick['share'] == 1:
            if not is_recording:
                print('Recording started...')
                record_start_time = time.monotonic()
            is_recording = True

        if is_recording:
            if time.monotonic() - record_start_time > RECORDING_DELAY:
                is_recording = False
                dcM.saveLog()
            else:
                img = wM.getImg(True, size=[240, 120])
                dcM.saveData(img, steering)

        motor.move(throttle, -steering)
        cv2.waitKey(1)


if __name__ == '__main__':
    main()
