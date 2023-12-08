import numpy as np
import screen_brightness_control as sbc
import cv2
import mediapipe as mp

# Initialize MediaPipe Hands model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
draw = mp.solutions.drawing_utils

# Access the camera
cap = cv2.VideoCapture(0)  # Use the default camera (0) or change the index if multiple cameras are available

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        break

    # Flip the frame horizontally for a mirrored view
    frame = cv2.flip(frame, 1)

    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect hand landmarks
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Get the thumb and index finger landmarks (landmark indexes: 4 and 8)
            thumb_tip = hand_landmarks.landmark[4]
            index_tip = hand_landmarks.landmark[8]

            # Calculate the distance between thumb and index finger (in pixels)
            thumb_x, thumb_y = int(thumb_tip.x * frame.shape[1]), int(thumb_tip.y * frame.shape[0])
            index_x, index_y = int(index_tip.x * frame.shape[1]), int(index_tip.y * frame.shape[0])

            # Calculate Euclidean distance between thumb and index finger
            distance = np.sqrt((index_x - thumb_x)**2 + (index_y - thumb_y)**2)

            # Map distance to brightness level (adjust these values according to your needs)
            brightness_level = np.interp(distance, [50, 300], [0, 100])

            # Set brightness using screen_brightness_control library
            sbc.set_brightness(int(brightness_level))

            # Visualize hand landmarks on the frame
            draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Display the frame
    cv2.imshow('Hand Movement Detection', frame)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the OpenCV window
cap.release()
cv2.destroyAllWindows()
