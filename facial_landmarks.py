# https://pysource.com/blur-faces-in-real-time-with-opencv-mediapipe-and-python
import mediapipe as mp
import cv2
import numpy as np


class FaceLandmarks:
    def __init__(self):
        mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = mp_face_mesh.FaceMesh()

    def get_facial_landmarks(self, frame):
        mouth = np.array([
            164,
            393,
            391,
            322,
            410,
            287,
            273,
            335,
            406,
            313,
            18,
            83,
            182,
            106,
            43,
            57,
            186,
            92,
            165,
            167,
            164
        ])
        height, width, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.face_mesh.process(frame_rgb)

        facelandmarks = []
        if result.multi_face_landmarks:
            for facial_landmarks in result.multi_face_landmarks:
                for i in mouth:
                    pt1 = facial_landmarks.landmark[i]
                    x = int(pt1.x * width)
                    y = int(pt1.y * height)
                    facelandmarks.append([x, y])
        return np.array(facelandmarks, np.int32)

    def get_lips_landmarks(self, frame):
        mouth = np.array([
            0,
            267,
            269,
            270,
            409,
            291,
            375,
            321,
            405,
            314,
            17,
            84,
            181,
            91,
            146,
            61,
            185,
            40,
            39,
            37,
        ])
        height, width, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.face_mesh.process(frame_rgb)

        facelandmarks = []
        if result.multi_face_landmarks:
            for facial_landmarks in result.multi_face_landmarks:
                for i in mouth:
                    pt1 = facial_landmarks.landmark[i]
                    x = int(pt1.x * width)
                    y = int(pt1.y * height)
                    facelandmarks.append([x, y])
        return np.array(facelandmarks, np.int32)

    def get_mask_landmarks(self, frame):
        mouth = np.array([
            164,
            393,
            391,
            322,
            436,
            434,
            430,
            431,
            262,
            428,
            199,
            208,
            32,
            211,
            210,
            214,
            216,
            165,
            92,
            165,
            167,
            164,
        ])
        height, width, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.face_mesh.process(frame_rgb)

        facelandmarks = []
        if result.multi_face_landmarks:
            for facial_landmarks in result.multi_face_landmarks:
                for i in mouth:
                    pt1 = facial_landmarks.landmark[i]
                    if i == 322:
                        x = int(pt1.x * width + 20)
                        y = int(pt1.y * height - 20)
                    elif i == 92:
                        x = int(pt1.x * width - 20)
                        y = int(pt1.y * height - 20)
                    else:
                        x = int(pt1.x * width)
                        y = int(pt1.y * height)
                    facelandmarks.append([x, y])
        return np.array(facelandmarks, np.int32)
