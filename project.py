import cv2 as cv
import numpy as np
import glob
from CalibrationInstance import CalibrationInstance

# Stride length
stride = 44 # mm

# Termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points on a 7x7 grid, like (0,0,0), (1,0,0), (2,0,0) ....,(7,7,0)
objp = np.zeros((7*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:7].T.reshape(-1,2) * stride
axis_points = np.float32([[stride*4,0,0], [0,stride*4,0], [0,0,-stride*4]]).reshape(-1,3)
 
# Arrays to store object points and image points for calibration from all images
objpoints = [] # 3d points in real world space
imgpoints_25 = {} # 2d points in image plane.
imgpoints_10 = {}
imgpoints_5 = {}
 
# Sort test images
images = sorted(glob.glob('media/*.jpeg'))

# Stores images that have successfully passed corner detection
images_25 = images.copy()
images_10 = []
images_5 = []

flags = cv.CALIB_CB_EXHAUSTIVE + cv.CALIB_CB_ACCURACY

# Initialize calibration variables
ret, matrix, distortion_coef, rotation_vecs, translation_vecs = None, None, None, None, None

clicked_points = []

# Function called upon mouse click
def mouse_callback(event, x, y, flags, param):
    global clicked_points, img_display
    if event == cv.EVENT_LBUTTONDOWN:
        # Register the click
        clicked_points.append((x, y))
        # Draw a filled circle at the clicked point (red, radius 5)
        cv.circle(img_display, (x, y), 5, (0, 0, 255), -1)
        cv.imshow('img', img_display)

# Print coordinates on mouse click
def manual_check(fname):
   global clicked_points, img_display
   clicked_points = []  # Reset click storage

   print("Manual check: please click:")
   print("Top-left (but the bottom right of this square),")
   print("Top-right (etc, should give a 6x6 instead of 8x8 grid),")
   print("Bottom-right,")
   print("Bottom-left.")
   img = cv.imread(fname)

   # Create a copy for displaying the clicks
   img_display = img.copy()

   cv.imshow('img', img_display)
   cv.setMouseCallback('img', mouse_callback)

   # Wait until 4 clicks have been collected or ESC is pressed.
   while len(clicked_points) < 4:
      if cv.waitKey(20) & 0xFF == 27:  # Exit if ESC key is pressed
          break

   cv.destroyAllWindows()
   return np.array(clicked_points, dtype=np.float32).reshape(-1, 1, 2)

# Calculates the distance from the world origin to the camera
def distance_to_camera():
    global matrix, distortion_coef

    # Checks for calibration errors
    if not objpoints or not imgpoints_25:
      print("Error! Image or object points missing.")
      return None, None
    if matrix is None or distortion_coef is None:
      print("Error! Matrix or distortion coefficent missing.")
      return None, None

    # Gets rotation and translation vectors from solvePNP
    success, rvec, tvec = cv.solvePnP(objpoints[0], imgpoints_25[images_25[0]], matrix, distortion_coef) # objpoints[0]?

    if success:
      # Find and print distance from origin to camera
      distance = np.linalg.norm(tvec)

      print(f"Distance to camera: {distance:.2f} mm")
      return rvec, tvec

    else:
      print("Error! SolvePNP failed.")
      return None, None
    
# CHOICE TASK
# Preprocesses the image for clarity, increasing corner detection rate
def preprocessing(img):

    # Convert image to grayscale
    processed_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Apply CLAHE preprocessing for increased contrast and glare reduction
    clahe = cv.createCLAHE(clipLimit=20.0, tileGridSize=(8, 8))
    processed_img = clahe.apply(processed_img)

    # Apply slight gaussian blur to reduce the effect of glare on edge detection
    processed_img = cv.GaussianBlur(processed_img, (5, 5), 0)

    return processed_img

# Retrieve 2D and 3D points from chessboard images
def get_points():
    found = 0 #stores how many images are successfully processed

    for fname in images:
      print(fname)
      img = cv.imread(fname)

      # Resize img (needs to be resized in projection as well if we decide to do so)
      #img = cv.resize(img, (int(img.shape[1]/2), int(img.shape[0]/2))) 

      # Preprocess each image to increase edge detection
      preprocessed = preprocessing(img)

      # Find the chess board corners
      # https://stackoverflow.com/a/76833504/24809902 for flags
      
      ret, corners = cv.findChessboardCornersSB(preprocessed, (7,7), flags=flags)

      # If found, add object points, image points (after refining them)
      if ret:
          print("found")
          objpoints.append(objp) # correct?
          found += 1

          refined_corners = cv.cornerSubPix(preprocessed, corners, (11,11), (-1,-1), criteria)
          imgpoints_25[fname] = refined_corners
          if not (found >= 10):
              imgpoints_10[fname] = refined_corners
              images_10.append(fname)
              if not (found >= 5):
                  imgpoints_5[fname] = refined_corners
                  images_5.append(fname)

          # Draw and display the corners
          cv.drawChessboardCorners(img, (7,7), refined_corners, ret)
          cv.imshow('img', img)
          cv.waitKey(500)
      else:
          print("not found")
          interpolated_corners = manual_check(fname)

          # Use 2D ideal grid corners for homography
          # Define 4 corners of the ideal 7x7 grid (2D!)
          ideal_manual_points = np.array([
              [0, 0],    # top-left
              [6, 0],    # top-right (7x7 grid has 0-6 in x)
              [6, 6],    # bottom-right
              [0, 6]     # bottom-left
          ], dtype=np.float32).reshape(-1, 1, 2)

          # Compute homography
          H, _ = cv.findHomography(ideal_manual_points, interpolated_corners)

          x, y = np.meshgrid(np.arange(7), np.arange(7))

          ideal_grid = np.float32(np.vstack([x.ravel(), y.ravel()]).T).reshape(-1, 1, 2)

          interpolated_corners = cv.perspectiveTransform(ideal_grid, H).reshape(-1, 2)

          # Register the points
          objpoints.append(objp)  # 3D object points
          imgpoints_25[fname] = interpolated_corners
          found += 1

          # Draw and display the corners
          draw_img = img.copy()
          cv.drawChessboardCorners(draw_img, (7,7), interpolated_corners, True)
          cv.imshow('img', draw_img)
          cv.waitKey(500)
        
    print("\nSuccessfully processed " + str(found) + " images.")
    cv.destroyAllWindows()


# 25 images
# 10 automatically detected images
# 5 out of the 10 from run 2
# -> 3x of (ret, matrix, distortion_coef, rotation_vecs, translation_vecs)

def calibrate_camera(imgpoints):
    global matrix, distortion_coef, rotation_vecs, translation_vecs
    # Preprocesses the calibration image
    image = cv.imread(images[0])
    preprocessed = preprocessing(image)

    # Caibrate camera
    # objpoints?
    ret, matrix, distortion_coef, rotation_vecs, translation_vecs = cv.calibrateCamera(objpoints, list(imgpoints.values()), preprocessed.shape[::-1], None, None)
    return CalibrationInstance(ret, matrix, distortion_coef, rotation_vecs, translation_vecs)

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

def project_cube(webcam=False):
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
    calibration_25 = calibrate_camera(imgpoints_25)
    calibration_10 = calibrate_camera(imgpoints_25)
    calibration_5 = calibrate_camera(imgpoints_25)
    rotation_vecs = calibration_25.rotation_vecs
    translation_vecs = calibration_25.translation_vecs
    matrix = calibration_25.matrix
    distortion_coef = calibration_25.distortion_coef

    distance_to_camera()

    test_idx = 1

    if (webcam):
        while True:
            # Initialize webcam
            cap = cv.VideoCapture(0)
            ret, frame = cap.read()
            if not ret:
              break

            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            found, corners = cv.findChessboardCornersSB(gray, (7,7), flags=flags)

            if found:
                # Solve PnP
                ret, rvec, tvec = cv.solvePnP(objp, corners, matrix, distortion_coef)
        
                if ret:
                    # Project cube
                    cube_proj, _ = cv.projectPoints(cube_points, rvec, tvec, matrix, distortion_coef)
                    cube_proj = cube_proj.reshape(-1, 2).astype(int)
                    
                    # Draw cube
                    edges = [(0,1),(1,2),(2,3),(3,0),
                            (4,5),(5,6),(6,7),(7,4),
                            (0,4),(1,5),(2,6),(3,7)]
                    for s, e in edges:
                        cv.line(frame, tuple(cube_proj[s]), tuple(cube_proj[e]), (0,255,0), 2)
            cv.imshow('AR Cube Projection', frame)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv.destroyAllWindows()
    
    else:

      fname = images_25[test_idx]
      img = cv.imread(fname)
      gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
      corners2 = cv.cornerSubPix(gray, imgpoints_25[fname], (11,11), (-1,-1), criteria)
      ret, rotation_vecs, translation_vecs = cv.solvePnP(objp, corners2, matrix, distortion_coef)
      axis_imgpts, jac = cv.projectPoints(axis_points, rotation_vecs, translation_vecs, matrix, distortion_coef)
      cube_imgpts, jac = cv.projectPoints(cube_points, rotation_vecs, translation_vecs, matrix, distortion_coef)

      img = draw_axis(img, corners2, axis_imgpts)
      img = draw_cube(img, corners2, cube_imgpts)

      # Draw the chessboard corners on the image (using the first detected corners).
      img = cv.drawChessboardCorners(img, (7,7), imgpoints_25[images_25[test_idx]], True)

      cv.imshow('img', img)
      k = cv.waitKey(0) & 0xFF
      if k == ord('s'):
        cv.imwrite(fname[:6]+'.png', img)