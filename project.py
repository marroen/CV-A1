import cv2 as cv
import numpy as np
import glob
from CalibrationInstance import CalibrationInstance
import util

# Stride length
stride = 44 # mm

# Termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points on a 7x7 grid, like (0,0,0), (1,0,0), (2,0,0) ....,(7,7,0)
objp = np.zeros((7*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:7].T.reshape(-1,2) * stride
axis_points = np.float32([[stride*4,0,0], [0,stride*4,0], [0,0,-stride*4]]).reshape(-1,3)
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
 
# Arrays to store object points and image points for calibration from all images
points_25 = {} # 'image_fname: (objpoints, imgpoints)
points_10 = {}
points_5  = {}
 
# Sort test images
images = sorted(glob.glob('media/*.jpg') + glob.glob('media/*.jpeg'))

# Flags for corner detection
flags = cv.CALIB_CB_EXHAUSTIVE + cv.CALIB_CB_ACCURACY

# Initialize calibration variables
ret, matrix, distortion_coef, rotation_vecs, translation_vecs = None, None, None, None, None

clicked_points = []

# Function called upon mouse click
def mouse_callback(event, x, y, flags, param):
    global clicked_points
    if event == cv.EVENT_LBUTTONDOWN:
        if len(clicked_points) < 4:
            clicked_points.append((x, y))
            print(f"Clicked point {len(clicked_points)}: ({x}, {y})")

# Save corner coordinates through mouse clicks (including choice task for linear interpolation improvement)
def manual_check(img):
    global clicked_points, img_display, warped_display
    clicked_points = []  # Reset click storage

    util.print_manual_calibration_guide()

    # Create a copy for displaying the clicks
    img_display = img.copy()

    cv.namedWindow('Original Perspective')
    cv.setMouseCallback('Original Perspective', mouse_callback)
    
    while True:
        # Create temp image with current points
        temp_img = img.copy()
        for pt in clicked_points:
            cv.circle(temp_img, tuple(pt), 5, (0, 0, 255), -1)
        
        cv.imshow('Original Perspective', temp_img)
        key = cv.waitKey(20)
        
        if len(clicked_points) >= 4:
            break
        if key == 27:  # ESC pressed
            cv.destroyAllWindows()
            return None
    
    cv.destroyAllWindows()

    # CHOICE TASK
    # Create warped view for improving linear interpolation
    src_points = np.array(clicked_points, dtype=np.float32)
    warped_size = 600
    dst_points = np.array([[0,0], [warped_size,0], 
                         [warped_size,warped_size], [0,warped_size]], 
                        dtype=np.float32)
    M = cv.getPerspectiveTransform(src_points, dst_points)
    warped = cv.warpPerspective(img, M, (warped_size, warped_size))
    
    # Second stage: warped perspective
    clicked_points = []
    warped_display = warped.copy()
    
    cv.namedWindow('Warped Perspective')
    cv.setMouseCallback('Warped Perspective', mouse_callback)
    while True:
        cv.imshow('Warped Perspective', warped_display)
        key = cv.waitKey(20)
        if len(clicked_points) >= 4 or key == 27:
            break
    cv.destroyAllWindows()
    
    if len(clicked_points) < 4:
        return None

    # Back-project points
    warped_points = np.array(clicked_points, dtype=np.float32).reshape(-1, 1, 2)
    _, Minv = cv.invert(M)
    original_points = cv.perspectiveTransform(warped_points, Minv)
    
    return original_points.reshape(-1, 1, 2)

#finds the roll, pitch, and yaw of the origin with respect to the camera
def find_roll_pitch_yaw(R):

    sy = np.sqrt(R[0, 0] ** 2 + R[0, 1] ** 2)

    roll = np.arctan2(-R[1, 2], R[2, 2])
    pitch = np.arctan2(R[0, 2], sy)
    yaw = np.arctan2(R[0, 1], R[0, 0])

    return np.degrees([roll, pitch, yaw])  # Return values in degrees

# Calculates the distance from the world origin to the camera
def solvepnp_vectors(fname):
    global matrix, distortion_coef

    # Gets rotation and translation vectors from solvePNP
    success, rvec, tvec = cv.solvePnP(points_25[fname][0], points_25[fname][1], matrix, distortion_coef)

    if success:
      return rvec, tvec
    else:
      print("Error! SolvePNP failed.")
      return None, None, None

# CHOICE TASK
# Preprocesses the image for clarity, increasing corner detection rate
def preprocessing(img):

    # Resize img (needs to be resized in projection as well if we decide to do so)
    processed_img = cv.resize(img, (int(img.shape[1]/2), int(img.shape[0]/2)))

    # Convert image to grayscale
    processed_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Apply CLAHE preprocessing for increased contrast and glare reduction
    clahe = cv.createCLAHE(clipLimit=20.0, tileGridSize=(8, 8))
    processed_img = clahe.apply(processed_img)

    # Apply slight gaussian blur to reduce the effect of glare on edge detection
    processed_img = cv.GaussianBlur(processed_img, (5, 5), 0)

    return processed_img

# Retrieve 2D and 3D points from chessboard images
def get_all_points(run):
    found = 0 #stores how many images are successfully processed

    for fname in images:
        _, _, new_found, _ = get_points(run, fname, found)
        # Handling skipped images
        if new_found > found:
            found = new_found

        # Once enough images have been processed for run, stop
        if new_found >= run:
            break
        
    print("\nSuccessfully processed " + str(found) + " images.")
    cv.destroyAllWindows()

# Retrieve 2D and 3D points from chessboard images
def get_points(run, fname, found):
    points_for_fname = (np.empty(0), np.empty(0), 0, None)
    print(fname)
    img = cv.imread(fname)

    # Preprocess each image to increase edge detection
    preprocessed = preprocessing(img)

    # Find the chess board corners  
    ret, corners = cv.findChessboardCornersSB(preprocessed, (7,7), flags=flags)

    # If found, add object points, image points (after refining them)
    if ret:
        print("found")
        found += 1

        refined_corners = cv.cornerSubPix(preprocessed, corners, (11,11), (-1,-1), criteria)
        points_for_fname = (objp, refined_corners, found, img)
        points_25[fname] = (objp, refined_corners)
        if (found <= 10):
            points_10[fname] = (objp, refined_corners)
            if (found <= 5):
                points_5[fname] = (objp, refined_corners)

        # Draw and display the corners
        draw_img = img.copy()
        cv.drawChessboardCorners(draw_img, (7,7), refined_corners, ret)
        cv.imshow('img', draw_img)
        cv.waitKey(500)
    else:
        print("not found")
        if (run != 25):
            return points_for_fname
        interpolated_corners = manual_check(preprocessed)

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
        points_25[fname] = (objp, interpolated_corners)
        points_for_fname = (objp, interpolated_corners, found, img)

        # Draw and display the corners
        draw_img = img.copy()
        cv.drawChessboardCorners(draw_img, (7,7), interpolated_corners, True)
        cv.imshow('img', draw_img)
        cv.waitKey(500)
    return points_for_fname


# Calibrate camera with given image points, 2D and 3D
def calibrate_camera(points):
    global matrix, distortion_coef, rotation_vecs, translation_vecs
    
    # Preprocesses the calibration image
    image = cv.imread(images[0])
    preprocessed = preprocessing(image)

    # Combine all object points and image points from the dictionary into lists.
    all_objpoints = [v[0] for v in points.values()]
    all_imgpoints = [v[1] for v in points.values()]

    # Caibrate camera
    ret, matrix, distortion_coef, rotation_vecs, translation_vecs = cv.calibrateCamera(all_objpoints, all_imgpoints, preprocessed.shape[::-1], None, None)
    print("Intrinsic Camera Matrix (K):\n", matrix)

    return CalibrationInstance(ret, matrix, distortion_coef, rotation_vecs, translation_vecs)

# Calculate axis coordinates and return image with it drawn
def draw_axis(img, corners, imgpts):
    corner = tuple(corners[0].ravel().astype("int32"))
    imgpts = imgpts.astype("int32")
    img = cv.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
    img = cv.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
    img = cv.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
    return img

# Calculate cube coordinates and return image with it drawn
def draw_cube(img, corners, imgpts, fname, dist, orient=0, rot=255):
    imgpts = np.int32(imgpts).reshape(-1,2)

    #gets the rotation vectors from solvepnp
    rvec, _ = solvepnp_vectors(fname)

    # Converts the rotation vectors to an object rotation matrix
    R, _ = cv.Rodrigues(rvec)

    roll, pitch, yaw = find_roll_pitch_yaw(R)
    #print("Roll, Pitch, Yaw):", roll, pitch, yaw)

    # Maps the values for yaw, pitch, and distance to HSV colorspace
    H = int(179 * ((yaw + 90) / 180)) if -90 <= yaw <= 90 else (0 if yaw < -90 else 179)
    S = int(255 * (1 - (pitch / 45))) if pitch < 45 else 0
    V = int(255 * (1 - (dist / 4000))) if dist <= 4000 else 0

    # Ensures HSV values are in a valid range
    H = np.clip(H, 0, 179)
    S = np.clip(S, 0, 255)
    V = np.clip(V, 0, 255)

    # Convert HSV to BGR
    hsv_color = np.uint8([[[H, S, V]]])
    bgr_color = cv.cvtColor(hsv_color, cv.COLOR_HSV2BGR)[0][0]
    B, G, R = map(int, bgr_color)

    # base
    img = cv.drawContours(img, [imgpts[:4]], -1, (0,255,0), 1)

    # walls
    for i, j in zip(range(4), range(4,8)):
      img = cv.line(img, tuple(imgpts[i]), tuple(imgpts[j]), (255), 1)

    # top
    img = cv.drawContours(img, [imgpts[4:]], -1, (B,G,R), -1)

    return img

# Project cube onto chessboard image
def project_cube(run, webcam=False):
    print("projecting")
    test_idx = 0
    
    calibration = None
    if run == 25:
        calibration = calibrate_camera(points_25)
    elif run == 10:
        calibration = calibrate_camera(points_10)
    elif run == 5:
        calibration = calibrate_camera(points_5)
    else:
        print("Unsupported")
        return
    
    # Set calibration variables
    matrix = calibration.matrix
    distortion_coef = calibration.distortion_coef
    rotation_vecs = calibration.rotation_vecs
    translation_vecs = calibration.translation_vecs

    # CHOICE TASK
    # Real-time projection of the cube with webcam
    if (webcam):
        while True:
            # Initialize webcam
            cap = cv.VideoCapture(0)
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv.resize(frame, (int(frame.shape[1]/2), int(frame.shape[0]/2)))
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
    
    # Non-webcam projection (on test image)
    else:
      # Sort test images
      test_images = sorted(glob.glob('media/tests/*.jpg'))

      print(len(points_25))
      print(len(points_10))
      print(len(points_5))

      fname = test_images[test_idx]

      # Automatically detect corners
      test_points = get_points(run, fname, 0)
      if not test_points[1].any():
        print("Error! Not automatically detected.")
        return

      # Find and print distance from origin to camera
      pnp_rvec, pnp_tvec = solvepnp_vectors(fname)
      dist = np.linalg.norm(pnp_tvec)
      print(f"Distance to camera: {dist:.2f} mm")

      img = test_points[3] # Same img (halved) from where points are initially extracted
      preprocessed = preprocessing(img)

      refined_corners = cv.cornerSubPix(preprocessed, test_points[1], (11,11), (-1,-1), criteria)

      ret, rvec, tvec = cv.solvePnP(objp, refined_corners, matrix, distortion_coef)
      axis_imgpts, _ = cv.projectPoints(axis_points, rvec, tvec, matrix, distortion_coef)
      cube_imgpts, _ = cv.projectPoints(cube_points, rvec, tvec, matrix, distortion_coef)

      img = draw_axis(img, refined_corners, axis_imgpts)
      img = draw_cube(img, refined_corners, cube_imgpts, fname, dist)

      # Draw the chessboard corners on the image (using the first detected corners).
      img = cv.drawChessboardCorners(img, (7,7), test_points[1], True)

      cv.imshow('img', img)
      k = cv.waitKey(0) & 0xFF
      if k == ord('s'):
        cv.imwrite(fname[:6]+'.png', img)