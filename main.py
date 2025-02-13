import cv2 as cv
import numpy as np
import glob

# Stride length
stride = 44 # mm

# Termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(7,7,0)
objp = np.zeros((7*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:7].T.reshape(-1,2) * stride
 
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
 
images = sorted(glob.glob('media/*.jpeg'))

images_final = images.copy()
gray_final = cv.imread(images[0], cv.IMREAD_GRAYSCALE)

ret, matrix, distortion_coef, rotation_vecs, translation_vecs = None, None, None, None, None

# Function called upon mouse click
def mouse_callback(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        print(x, y)

# Print coordinates on mouse click
def manual_check():
   print("Manual check")
   img = cv.imread(images[20])
   cv.imshow('img', img)
   
   cv.setMouseCallback('img', mouse_callback)
   cv.waitKey(0)

#manual_check()
  
# Retrieve 2D and 3D points from chessboard images
def get_points():
  for fname in images:
    print(fname)
    img = cv.imread(fname)
    # resize img
    #img = cv.resize(img, (int(img.shape[1]/2), int(img.shape[0]/2)))
    gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    gray_final = gray
    cv.imshow('gray', gray)

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
      images_final.remove(fname)
      print("not found")

  cv.destroyAllWindows()

get_points()

def calibrate_camera():
   ret, matrix, distortion_coef, rotation_vecs, translation_vecs = cv.calibrateCamera(objpoints, imgpoints, gray_final.shape[::-1], None, None)

# Function to draw a cube on the image
def draw_cube(img, imgpts):
    imgpts = np.int32(imgpts).reshape(-1, 2)

    # Draw base in green.
    img = cv.drawContours(img, [imgpts[:4]], -1, (0, 255, 0), 3)

    # Draw vertical edges in blue.
    for i, j in zip(range(4), range(4, 8)):
        img = cv.line(img, tuple(imgpts[i]), tuple(imgpts[j]), (255, 0, 0), 3)

    # Draw top in red.
    #img = cv.drawContours(img, [imgpts[4:]], -1, (0, 0, 255), 3)
    img = cv.fillConvexPoly(img, imgpts[4:], (0, 0, 255))
    return img

# Projecting cube
def project_cube():
  # If calibration succeeded and we have at least one chessboard pose, project and draw the cube.
  if len(rotation_vecs) > 0:
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
      img = cv.imread(images_final[0])
      img = cv.resize(img, (int(img.shape[1]/2), int(img.shape[0]/2)))

      # Draw the chessboard corners on the image (using the first detected corners).
      img = cv.drawChessboardCorners(img, (7,7), imgpoints[0], True)
      
      # Use the first pose (rvec and tvec) from calibration.
      imgpts, _ = cv.projectPoints(cube_points, rotation_vecs[0], translation_vecs[0], matrix, distortion_coef)
      
      # Draw the cube on the image.
      img = draw_cube(img, imgpts)
      
      cv.imshow('Cube Projection', img)
      cv.waitKey(0)
      cv.destroyAllWindows()