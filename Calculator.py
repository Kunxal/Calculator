import cv2
import mediapipe as mp
import numpy as np
import math

# Initialize MediaPipe Hand
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

# Initialize OpenCV window and canvas
canvas = np.zeros((480, 640, 3), dtype="uint8")
cap = cv2.VideoCapture(0)

# Variables to store finger drawing
is_drawing = False
prev_x, prev_y = None, None
draw_color = (255, 255, 255)

# Gesture recognition and drawing
while cap.isOpened():
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get index finger tip coordinates
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            x = int(index_finger_tip.x * frame.shape[1])
            y = int(index_finger_tip.y * frame.shape[0])

            # If the user is writing (index finger is close to the screen)
            if hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].z < -0.1:
                if prev_x is None and prev_y is None:
                    prev_x, prev_y = x, y

                is_drawing = True
                cv2.line(canvas, (prev_x, prev_y), (x, y), draw_color, 5)
                prev_x, prev_y = x, y
            else:
                is_drawing = False
                prev_x, prev_y = None, None

    # Combine the drawing with the webcam feed
    combined = cv2.addWeighted(frame, 0.5, canvas, 0.5, 0)

    # Display instructions
    cv2.putText(combined, "Use your index finger to draw and write equations", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Display the combined image
    cv2.imshow('Gesture Controlled Calculator', combined)

    if cv2.waitKey(1) & 0xFF == ord('c'):
        canvas = np.zeros((480, 640, 3), dtype="uint8")  # Clear the canvas
    elif cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
