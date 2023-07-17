# opencv-filter
        for i in range(len(landmarks_copy)):
            if i != 5 and i != 15:
                if 0 <= i <= 10:
                    landmarks_copy[i][0] = landmarks_copy[i][0]+10
                    landmarks_copy[i][1] = landmarks_copy[i][1]+10