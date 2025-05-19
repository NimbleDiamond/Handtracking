# PosetrackingZY.py

import cv2
import mediapipe as mp

def ZtoXTrack():
    # Initialize MediaPipe
    mp_pose = mp.solutions.pose
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    # Create instances
    pose = mp_pose.Pose(static_image_mode=False,
                        min_detection_confidence=0.7,
                        min_tracking_confidence=0.7)
    hands = mp_hands.Hands(static_image_mode=False,
                           max_num_hands=2,
                           min_detection_confidence=0.7,
                           min_tracking_confidence=0.7)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error opening camera.")
        return

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = pose.process(rgb_frame)
        hand_results = hands.process(rgb_frame)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            landmarks = results.pose_landmarks.landmark
            right_index_x = landmarks[mp_pose.PoseLandmark.RIGHT_INDEX].x
            right_index_y = landmarks[mp_pose.PoseLandmark.RIGHT_INDEX].y
            right_index_visibility = landmarks[mp_pose.PoseLandmark.RIGHT_INDEX].visibility

            left_index_x = landmarks[mp_pose.PoseLandmark.LEFT_INDEX].x
            left_index_y = landmarks[mp_pose.PoseLandmark.LEFT_INDEX].y
            left_index_visibility = landmarks[mp_pose.PoseLandmark.LEFT_INDEX].visibility

            index_values = [right_index_x, right_index_y, right_index_visibility,
                            left_index_x, left_index_y, left_index_visibility]

            print(index_values)  # Use or return this as needed

        cv2.imshow("Pose Tracking", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC key to exit
            break

    cap.release()
    cv2.destroyAllWindows()

