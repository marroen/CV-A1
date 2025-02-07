import cv2 as cv
import numpy as np
import glob

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((7*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:7].T.reshape(-1,2)
 
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
 
images = glob.glob('media/*.jpeg')

for fname in images:
  print(fname)
  img = cv.imread(fname)
  img = cv.resize(img, (int(img.shape[1]/2), int(img.shape[0]/2)))
  gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
  #cv.imshow('gray', img_gray)
  #cv.waitKey(500)

  # Find the chess board corners
  flags = cv.CALIB_CB_EXHAUSTIVE + cv.CALIB_CB_ACCURACY
  ret, corners = cv.findChessboardCornersSB(gray, (7,7), flags=flags)

  # If found, add object points, image points (after refining them)
  if ret == True:
    print("found")
    objpoints.append(objp)

    corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
    imgpoints.append(corners2)

    # Draw and display the corners
    cv.drawChessboardCorners(img, (7,7), corners2, ret)
    cv.imshow('img', img)
    cv.waitKey(500)
cv.destroyAllWindows()