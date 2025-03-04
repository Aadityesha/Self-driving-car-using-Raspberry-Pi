import cv2
import numpy as np

def thresholding(image: np.ndarray) -> np.ndarray:
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_white = np.array([80, 0, 0])
    upper_white = np.array([255, 160, 255])
    mask_white = cv2.inRange(image_hsv, lower_white, upper_white)
    return mask_white

def warp_image(image: np.ndarray, points: np.ndarray, width: int, height: int, inv: bool = False) -> np.ndarray:
    pts1 = np.float32(points)
    pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    if inv:
        matrix = cv2.getPerspectiveTransform(pts2, pts1)
    else:
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
    warped_image = cv2.warpPerspective(image, matrix, (width, height))
    return warped_image

def nothing(a: int) -> None:
    pass

def initialize_trackbars(initial_values: List[int], width: int = 480, height: int = 240) -> None:
    cv2.namedWindow("Trackbars")
    cv2.resizeWindow("Trackbars", 360, 240)
    cv2.createTrackbar("Width Top", "Trackbars", initial_values[0], width // 2, nothing)
    cv2.createTrackbar("Height Top", "Trackbars", initial_values[1], height, nothing)
    cv2.createTrackbar("Width Bottom", "Trackbars", initial_values[2], width // 2, nothing)
    cv2.createTrackbar("Height Bottom", "Trackbars", initial_values[3], height, nothing)

def val_trackbars(width: int = 480, height: int = 240) -> np.ndarray:
    width_top = cv2.getTrackbarPos("Width Top", "Trackbars")
    height_top = cv2.getTrackbarPos("Height Top", "Trackbars")
    width_bottom = cv2.getTrackbarPos("Width Bottom", "Trackbars")
    height_bottom = cv2.getTrackbarPos("Height Bottom", "Trackbars")
    points = np.float32([(width_top, height_top), (width - width_top, height_top),
                      (width_bottom , height_bottom), (width - width_bottom, height_bottom)])
    return
