import cv2
import mediapipe as mp
import pyautogui
import math
from collections import deque

screen_width, screen_height = pyautogui.size()

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)

cap = cv2.VideoCapture(0)

def euclidean_distance(point1, point2):
    return math.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)

# Store last positions for smoothing
positions_x = deque(maxlen=5)
positions_y = deque(maxlen=5)

# Sensitivity scale (lower = smaller movements)
SENSITIVITY = 1.2  # try 0.8 or 1.2 if needed
BLINK_THRESHOLD = 0.015

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Eye movement
            eye_landmark = face_landmarks.landmark[474]
            screen_x = int(eye_landmark.x * screen_width * SENSITIVITY)
            screen_y = int(eye_landmark.y * screen_height * SENSITIVITY)

            positions_x.append(screen_x)
            positions_y.append(screen_y)

            avg_x = int(sum(positions_x) / len(positions_x))
            avg_y = int(sum(positions_y) / len(positions_y))

            pyautogui.moveTo(avg_x, avg_y)

            # Blink detection
            top_lid = face_landmarks.landmark[386]
            bottom_lid = face_landmarks.landmark[374]
            dist = euclidean_distance(top_lid, bottom_lid)

            if dist < BLINK_THRESHOLD:
                pyautogui.click()
                pyautogui.sleep(0.25)

    cv2.imshow('Smooth Eye Mouse', frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
