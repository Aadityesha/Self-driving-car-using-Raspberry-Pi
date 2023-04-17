import pygame
from time import sleep

pygame.init()
controller = pygame.joystick.Joystick(0)
controller.init()

BUTTON_MAPPING = {
    0: 'x', 1: 'o', 2: 't', 3: 's',
    4: 'L1', 5: 'R1', 6: 'L2', 7: 'R2',
    8: 'share', 9: 'options'
}
AXIS_MAPPING = {
    0: 'axis1', 1: 'axis2', 2: 'axis3', 3: 'axis4'
}

def get_controller_input():
    pygame.event.pump()
    buttons = {}
    for button_idx, button_name in BUTTON_MAPPING.items():
        buttons[button_name] = controller.get_button(button_idx)
    for axis_idx, axis_name in AXIS_MAPPING.items():
        buttons[axis_name] = round(controller.get_axis(axis_idx), 2)
    return buttons

def main():
    buttons = get_controller_input()
    print(buttons['share'])
    sleep(0.05)

if __name__ == '__main__':
    while True:
        main()
