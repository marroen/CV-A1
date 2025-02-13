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
axis_points = np.float32([[stride*4,0,0], [0,stride*4,0], [0,0,-stride*4]]).reshape(-1,3)
 
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = {} # 2d points in image plane.
 
images = sorted(glob.glob('media/*.jpeg'))

images_final = images.copy()
gray_final = cv.imread(images[0], cv.IMREAD_GRAYSCALE)

ret, matrix, distortion_coef, rotation_vecs, translation_vecs = None, None, None, None, None

class CameraCalibration:
    def __init__(self, ret, matrix, distortion_coef, rotation_vecs, translation_vecs):
        self.ret = ret
        self.matrix = matrix
        self.distortion_coef = distortion_coef
        self.rotation_vecs = rotation_vecs
        self.translation_vecs = translation_vecs

    def project_points(self, object_points, pose_index=0):
        """
        Projects 3D object points to 2D image points using the specified calibration pose.

        :param object_points: np.array of 3D points to project.
        :param pose_index: index of the pose (rvec/tvec pair) to use.
        :return: projected 2D image points.
        """
        if not self.rotation_vecs or not self.translation_vecs:
            raise ValueError("No extrinsic parameters available.")

        imgpts, _ = cv.projectPoints(object_points,
                                     self.rotation_vecs[pose_index],
                                     self.translation_vecs[pose_index],
                                     self.matrix,
                                     self.distortion_coef)
        return imgpts

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
  
# Retrieve 2D and 3D points from chessboard images
def get_points():
  for fname in images:
    print(fname)
    img = cv.imread(fname)
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
      imgpoints[fname] = corners2

      # Draw and display the corners
      cv.drawChessboardCorners(img, (7,7), corners2, ret)
      cv.imshow('img', img)
      cv.waitKey(500)
    else:
      images_final.remove(fname)
      print("not found")

  cv.destroyAllWindows()

# 25 images
# 10 automatically detected images
# 5 out of the 10 from run 2
# -> 3x of (ret, matrix, distortion_coef, rotation_vecs, translation_vecs)

def calibrate_camera():
   ret, matrix, distortion_coef, rotation_vecs, translation_vecs = cv.calibrateCamera(objpoints, list(imgpoints.values()), gray_final.shape[::-1], None, None)
   return CameraCalibration(ret, matrix, distortion_coef, rotation_vecs, translation_vecs)

def draw_axis(img, corners, imgpts):
  corner = tuple(corners[0].ravel().astype("int32"))
  imgpts = imgpts.astype("int32")
  img = cv.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
  img = cv.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
  img = cv.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
  return img

def draw_cube(img, corners, imgpts):
  imgpts = np.int32(imgpts).reshape(-1,2)

  # base
  img = cv.drawContours(img, [imgpts[:4]], -1, (0,255,0), 1)

  # walls
  for i, j in zip(range(4), range(4,8)):
    img = cv.line(img, tuple(imgpts[i]), tuple(imgpts[j]), (255), 1)

  # top
  img = cv.drawContours(img, [imgpts[4:]], -1, (0,0,255), -1)

  return img

def project_cube():
   print("projecting")
   cube_points = np.float32([
          [0         , 0         , 0          ],
          [2 * stride, 0         , 0          ],
          [2 * stride, 2 * stride, 0          ],
          [0         , 2 * stride, 0          ],
          [0         , 0         , -2 * stride],
          [2 * stride, 0         , -2 * stride],
          [2 * stride, 2 * stride, -2 * stride],
          [0         , 2 * stride, -2 * stride]
      ])
   calibration = calibrate_camera()
   rotation_vecs = calibration.rotation_vecs
   translation_vecs = calibration.translation_vecs
   matrix = calibration.matrix
   distortion_coef = calibration.distortion_coef

   fname = images_final[0]
   img = cv.imread(fname)
   gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
   corners2 = cv.cornerSubPix(gray, imgpoints[fname], (11,11), (-1,-1), criteria)
   ret, rotation_vecs, translation_vecs = cv.solvePnP(objp, corners2, matrix, distortion_coef)
   axis_imgpts, jac = cv.projectPoints(axis_points, rotation_vecs, translation_vecs, matrix, distortion_coef)
   cube_imgpts, jac = cv.projectPoints(cube_points, rotation_vecs, translation_vecs, matrix, distortion_coef)

   img = draw_axis(img, corners2, axis_imgpts)
   img = draw_cube(img, corners2, cube_imgpts)

   # Draw the chessboard corners on the image (using the first detected corners).
   img = cv.drawChessboardCorners(img, (7,7), imgpoints[images_final[0]], True)

   cv.imshow('img', img)
   k = cv.waitKey(0) & 0xFF
   if k == ord('s'):
      cv.imwrite(fname[:6]+'.png', img)