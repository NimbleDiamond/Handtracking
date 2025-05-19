import cv2
import mediapipe as mp

#Declare Coordinate Values



# Initialize Mediapipe Hand solution
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands

#Does tracky stuff
pose = mp_pose.Pose(static_image_mode=False,
                       min_detection_confidence=0.7,
                       min_tracking_confidence=0.7)

#hands = mp_hands.Hands(static_image_mode=False,
#                       max_num_hands=2,
#                       min_detection_confidence=0.7,
#                       min_tracking_confidence=0.7)


mp_drawing = mp.solutions.drawing_utils

#open the camera
cap = cv2.VideoCapture(0)


# error check to make sure the camera is open
if not cap.isOpened():
    print("Error")
    exit()


#Main loop
def ZtoXTrack(cap):
while True:

    #capture frame by frame from the camera
    success, frame = cap.read()
    if not success:
        break

    # Flip the frame horizontally 
    frame = cv2.flip(frame, 1)

    # Convert the frame color from BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the RGB frame with MediaPipe Hands
    results = pose.process(rgb_frame)
    #Handresult = hands.process(rgb_frame)

    if results.pose_landmarks:
        pose_landmarks = results.pose_landmarks
        # Draw landmarks
        mp_drawing.draw_landmarks(frame, pose_landmarks,mp_pose.POSE_CONNECTIONS)
    
    right_index_x = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_INDEX].x
    right_index_y = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_INDEX].y
    right_index_visibility = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_INDEX].visibility
    Left_index_x = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_INDEX].x
    Left_index_y = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_INDEX].y
    Left_index_visibility = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_INDEX].visibility

    IndexValues = [right_index_x, right_index_y, right_index_visibility, left_index_x, left_index_y, left_index_visibility]




#KILL THE CAMERA
cap.release()
cv2.destroyAllWindows()
