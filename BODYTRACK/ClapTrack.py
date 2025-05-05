import cv2
import time
import mediapipe as mp
import pyautogui as pag
import numpy as np




# Initialize Mediapipe Pose solution
mp_pose = mp.solutions.pose

pose = mp_pose.Pose(static_image_mode=False,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)

mp_drawing = mp.solutions.drawing_utils

#open the camera
cap = cv2.VideoCapture(0)


# error check to make sure the camera is open
if not cap.isOpened():
    print("Error")
    exit()


# Set the screen resolution (width, height)
screen_width, screen_height = pag.size()

mouseDown = False

#Main loop
while True:

    #capture frame by frame from the camera
    success, frame = cap.read()
    if not success:
        break

    # Flip the frame horizontally 
    frame = cv2.flip(frame, 1)

    # Convert the frame color from BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the RGB frame with MediaPipe Pose
    results = pose.process(rgb_frame)

    #frame resoulution
    frame_height, frame_width, _ = frame.shape

    if results.pose_landmarks:
        pose_landmarks = results.pose_landmarks
            # Draw landmarks
        mp_drawing.draw_landmarks(frame, pose_landmarks,mp_pose.POSE_CONNECTIONS)

        left_index = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_INDEX]
        right_index = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_INDEX]



            # Get the distance between the thumb and index finger
        distance = np.sqrt((left_index.x - right_index.x)**2 + (left_index.y - right_index.y)**2)

        if distance < 0.1:
            print("Clap")
            time.sleep(0.25)
            

        # Display the resulting frame
    cv2.imshow("Frame", frame)
    cv2.waitKey(1)


# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
