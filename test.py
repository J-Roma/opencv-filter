import cv2
import mediapipe as mp
import numpy as np

# Inicializar el video capturando la webcam
cap = cv2.VideoCapture(0)

# Cargar el modelo de puntos clave faciales de Mediapipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

while True:
    # Leer el siguiente frame del video
    ret, frame = cap.read()
    if not ret:
        break

    # Convertir el frame a RGB (mediapipe requiere imágenes en RGB)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detectar los puntos clave faciales en el frame
    results = face_mesh.process(frame_rgb)
    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]

        # Obtener las coordenadas de los puntos clave faciales de la boca
        mouth_landmarks = []
        for lm in face_landmarks.landmark:
            mouth_landmarks.append([int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])])

        # Convertir las coordenadas de los puntos clave a un arreglo numpy
        mouth_points = np.array(mouth_landmarks, np.int32)

    # Crear una máscara en blanco para la boca
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)

    # Dibujar la máscara de la boca utilizando los puntos de referencia
    cv2.drawContours(mask, [mouth_points], -1, 255, cv2.FILLED)

    # Aplicar la máscara en el frame original
    output_frame = cv2.bitwise_and(frame, frame, mask=mask)

    # Mostrar el frame resultante
    cv2.imshow("Video en tiempo real", output_frame)

    # Detener el bucle si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar los recursos
cap.release()
cv2.destroyAllWindows()