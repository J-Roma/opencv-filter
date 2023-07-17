import cv2
import mediapipe as mp
import numpy as np
from facial_landmarks import FaceLandmarks

# Load face landmarks
fl = FaceLandmarks()

cap = cv2.VideoCapture(0)


def zoom_at(img, zoom=1, angle=0, coord=None):
    cy, cx = [i / 2 for i in img.shape[:-1]] if coord is None else coord[::-1]

    rot_mat = cv2.getRotationMatrix2D((cx, cy), angle, zoom)
    get = cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR)

    return get


while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_copy = frame.copy()
    height, width, _ = frame.shape

    # Face landmarks detection
    landmarks = fl.get_lips_landmarks(frame)
    if len(landmarks) > 1:
        # Face Mask
        mask = np.zeros((height, width), np.uint8)
        cv2.fillConvexPoly(mask, landmarks, 255)
        # Center
        coordX = ((landmarks[5][0] + landmarks[15][0]) / 2)
        coordY = ((landmarks[0][1] + landmarks[10][1]) / 2)
        # Zoom Ratio
        scale = 2.1
        # Zoom Lips
        # frame_copy = zoom_at(frame_copy, scale, coord=(coordX, coordY))
        face_extracted = cv2.bitwise_and(frame_copy, frame_copy, mask=mask)
        face_extracted = zoom_at(face_extracted, scale, coord=(coordX, coordY))

        # Extract background
        mask = zoom_at(mask, scale, coord=(coordX, coordY))
        background_mask = cv2.bitwise_not(mask)
        background = cv2.bitwise_and(frame, frame, mask=background_mask)

        # Final result
        result = cv2.add(background, face_extracted)
        cv2.imshow("Result2", result)

    # cv2.imshow("Frame", frame)
    key = cv2.waitKey(30)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
