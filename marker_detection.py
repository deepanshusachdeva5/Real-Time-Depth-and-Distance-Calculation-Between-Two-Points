import cv2
import imutils
import sys
import os
import numpy as np
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to the image")
ap.add_argument("-t", "--type", required=True,
                help="tag of the marker to b detected")
args = vars(ap.parse_args())

image = cv2.imread(args['image'])

ARUCO_DICT = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
    "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
    "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
    "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
    "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
    "DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
    "DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
    "DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
    "DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
}


arucoDict = cv2.aruco.Dictionary_get(ARUCO_DICT[args['type']])
arucoParams = cv2.aruco.DetectorParameters_create()
(corners, ids, rejected) = cv2.aruco.detectMarkers(
    image, arucoDict, parameters=arucoParams)

if len(corners) > 0:
    ids = ids.flatten()
    for (markerCorner, markerId) in zip(corners, ids):

        corners_abcd = markerCorner.reshape((4, 2))
        (topLeft, topRight, bottomRight, bottomLeft) = corners_abcd

        topRightPoint = (int(topRight[0]), int(topRight[1]))
        topLeftPoint = (int(topLeft[0]), int(topLeft[1]))
        bottomRightPoint = (int(bottomRight[0]), int(bottomRight[1]))
        bottomLeftPoint = (int(bottomLeft[0]), int(bottomLeft[1]))

        cv2.line(image, topLeftPoint, topRightPoint, (0, 255, 0), 2)
        cv2.line(image, topRightPoint, bottomRightPoint, (0, 255, 0), 2)
        cv2.line(image, bottomRightPoint, bottomLeftPoint, (0, 255, 0), 2)
        cv2.line(image, bottomLeftPoint, topLeftPoint, (0, 255, 0), 2)

        cX = int((topLeft[0] + bottomRight[0])//2)
        cY = int((topLeft[1] + bottomRight[1])//2)
        cv2.circle(image, (cX, cY), 4, (255, 0, 0), -1)

        cv2.putText(image, str(
            int(markerId)), (int(topLeft[0]-10), int(topLeft[1]-10)), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255))

        # print(arucoDict)
        cv2.imshow("[INFO] marker detected", image)
        cv2.waitKey(0)

else:
    # print("[INFO] No marker Detected")
    pass

cv2.destroyAllWindows()
