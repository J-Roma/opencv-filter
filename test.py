import cv2
import mediapipe as mp
import numpy as np
from facial_landmarks import FaceLandmarks

class MouthZoomFilter:
    def __init__(self):
        self.fl = FaceLandmarks()
        self.scale = 2

    def zoom_at(self, img, zoom=1, angle=0, coord=None):
        cy, cx = [i / 2 for i in img.shape[:-1]] if coord is None else coord[::-1]
        rot_mat = cv2.getRotationMatrix2D((cx, cy), angle, zoom)
        return cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR)

    def apply_filter(self, frame):
        frame_copy = frame.copy()
        height, width, _ = frame.shape

        landmarks = self.fl.get_lips_landmarks(frame)
        if len(landmarks) > 1:
            mask = np.zeros((height, width), np.uint8)
            cv2.fillConvexPoly(mask, landmarks, 255)
            coordX = ((landmarks[5][0] + landmarks[15][0]) / 2)
            coordY = ((landmarks[0][1] + landmarks[10][1]) / 2)

            face_extracted = cv2.bitwise_and(frame_copy, frame_copy, mask=mask)
            face_extracted = self.zoom_at(face_extracted, self.scale, coord=(coordX, coordY))

            mask = self.zoom_at(mask, self.scale, coord=(coordX, coordY))
            background_mask = cv2.bitwise_not(mask)
            background = cv2.bitwise_and(frame, frame, mask=background_mask)

            result = cv2.add(background, face_extracted)
            return result
        else:
            return frame

cap = cv2.VideoCapture(0)
mouth_filter = MouthZoomFilter()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    filtered_frame = mouth_filter.apply_filter(frame)
    cv2.imshow("Result", filtered_frame)

    key = cv2.waitKey(30)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()