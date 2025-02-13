import cv2 as cv

# Class to store calibration data and project 3D points to 2D image points
class CalibrationInstance:
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