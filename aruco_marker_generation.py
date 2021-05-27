# python .\aruco_marker_generation.py --output tags_output --id 50 --type DICT_6X6_250 --padding 20
import cv2
import numpy as np
import argparse
import sys
import os

ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True,
                help="output path to store the image")
ap.add_argument("-i", "--id", type=int, required=True,
                help="id of the aruco marker to be genearted")
ap.add_argument("-t", "--type", type=str,
                default="DICT_ARUCO_ORIGINAL", help="type of ARUCO marker")
ap.add_argument("-p", "--padding", type=int,
                default=1, help="padding to the marker")
args = vars(ap.parse_args())

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

if ARUCO_DICT.get(args['type'], None) is None:
    print("[INFO] {} type of marker is not supported , please choose a different one".format(
        args['type']))
    sys.exit(0)

arucoDict = cv2.aruco.Dictionary_get(ARUCO_DICT[args['type']])

tag = np.zeros((300, 300, 1), dtype="uint8")
cv2.aruco.drawMarker(arucoDict, args["id"], 300, tag, args["padding"])

if not os.path.exists(args["output"] + "/"):
    os.makedirs(args["output"] + "/")

cv2.imwrite(args["output"] + "/" + args["type"] + "_" + str(args["padding"]) +
            "_" + str(args["id"]) + ".jpg", tag)
cv2.imshow("marker", tag)
cv2.waitKey(0)
cv2.destroyAllWindows()
