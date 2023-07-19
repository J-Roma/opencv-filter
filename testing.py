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
        # Create a mask for lips
        mask = np.zeros(frame_copy.shape[:2], dtype=np.uint8)
        cv2.fillConvexPoly(mask, landmarks, 255)
        face_extracted = cv2.bitwise_and(frame_copy, frame_copy, mask=mask)
        x, y, w, h = cv2.boundingRect(landmarks)
        center = (x+w//2, y+h//2)
        diag = (w**2 + h**2)**0.5
        ratio = int(diag*1/4)
        x, y, w, h = x - ratio, y - ratio, w + ratio, h + ratio
        cropped_lips = frame[y:y+h, x:x+w]
        mask = np.zeros(cropped_lips.shape[:-1])
        cv2.fillConvexPoly(mask, landmarks - np.array([x, y]), 255)
        cropped_lips = np.expand_dims(mask, 2).astype("bool")  * cropped_lips
        resized_lips = cv2.resize(cropped_lips, None, fx=2, fy=2)
        Real_thing = np.uint8(255 * (resized_lips > [0,0,0]))
        result = cv2.seamlessClone(np.uint8(resized_lips), frame, Real_thing, center, cv2.NORMAL_CLONE)
        cv2.imshow("Camera Window", result)
    else:
        cv2.imshow("Result", frame_copy)

    key = cv2.waitKey(30)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()


