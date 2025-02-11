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
  # resize img
  img = cv.resize(img, (int(img.shape[1]/2), int(img.shape[0]/2)))
  gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

  # Find the chess board corners
  # https://stackoverflow.com/a/76833504/24809902 for flags
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
  else:
    images.remove(fname)
    print("not found")

cv.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# Function to draw a cube on the image
def draw_cube(img, imgpts):
    imgpts = np.int32(imgpts).reshape(-1, 2)
    # Draw base in green.
    img = cv.drawContours(img, [imgpts[:4]], -1, (0, 255, 0), 3)
    # Draw vertical edges in blue.
    for i, j in zip(range(4), range(4, 8)):
        img = cv.line(img, tuple(imgpts[i]), tuple(imgpts[j]), (255, 0, 0), 3)
    # Draw top in red.
    img = cv.drawContours(img, [imgpts[4:]], -1, (0, 0, 255), 3)
    return img

# If calibration succeeded and we have at least one chessboard pose, project and draw the cube.
if len(rvecs) > 0:
    # Define the 3D cube points (origin at the chessboard corner).
    cube_points = np.float32([
        [0, 0, 0],
        [2, 0, 0],
        [2, 2, 0],
        [0, 2, 0],
        [0, 0, -2],
        [2, 0, -2],
        [2, 2, -2],
        [0, 2, -2]
    ])
    
    # Re-read the first image to draw the cube.
    img = cv.imread(images[0])
    img = cv.resize(img, (int(img.shape[1]/2), int(img.shape[0]/2)))

    # Draw the chessboard corners on the image (using the first detected corners).
    img = cv.drawChessboardCorners(img, (7,7), imgpoints[0], True)
    
    # Use the first pose (rvec and tvec) from calibration.
    imgpts, _ = cv.projectPoints(cube_points, rvecs[0], tvecs[0], mtx, dist)
    
    # Draw the cube on the image.
    img = draw_cube(img, imgpts)
    
    cv.imshow('Cube Projection', img)
    cv.waitKey(0)
    cv.destroyAllWindows()