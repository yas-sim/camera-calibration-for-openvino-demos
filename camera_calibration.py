import sys
import argparse

import numpy as np
import cv2

def main(args):
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    gx, gy = eval(args.grid_size)
    objp = np.zeros((gx*gy,3), np.float32)
    objp[:,:2] = np.mgrid[0:gx,0:gy].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    cap = cv2.VideoCapture(args.cam)
    if cap.isOpened() == False:
        print('Failed to open webcam({})'.format(args.cam))
        sys.exit(1)

    print('*** CAMERA CALIBRATION TOOL for OPENCV ***\n')
    print('Present {}x{} chess board pattern to the webCam'.format(gx, gy))
    print('The detected corners will be marked when the program detects the corners')
    print('Calibration may require >10 data to generate accurate camera matrix and distortion data')
    print('Keys:')
    print('  SPACE : Capture a chess board corner data')
    print('  \'c\'   : Generate the camera calibration data and exit the program')
    print('  ESC   : Exit the program')

    count = 0
    while True:
        ret, img = cap.read()
        if ret == False:
            break
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (gx,gy),None)

        # If found, add object points, image points (after refining them)
        if ret == True:
            corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (gx,gy), corners2,ret)

        cv2.imshow('img',img)
        key = cv2.waitKey(1)

        if key == 27:
            break
        if key == ord(' ') and ret==True:
            count += 1
            print('Captured : {}'.format(count))
            objpoints.append(objp)
            imgpoints.append(corners2)
        if key == ord('c'):
            print('Calculating...', end='', flush=True)
            rms, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
            # Save data
            print('RMS=', rms)
            print(mtx, dist)
            print('camera parameters are saved (\'{}.npz\')'.format(args.output))
            np.savez(args.output, mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-gs', '--grid_size', type=str, required=False, default='(10,7)', help='Default chess board grid size. default=(10,7)')
    parser.add_argument('-cam', type=int, required=False, default=0, help='WebCam #. default=0')
    parser.add_argument('-o', '--output', type=str, required=False, default='calib', help='Calibration data output file base name. default=\'calib\'')
    args = parser.parse_args()
    main(args)
