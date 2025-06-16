import cv2
import mediapipe as mp
#X = open("Zxis.txt","a")
#Main loop
def ZtoXTrack():
        #Loop value
    i = 0


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
    cap = cv2.VideoCapture(1)


    # error check to make sure the camera is open
    if not cap.isOpened():
        print("Error")
        exit()
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
                    #Draw landmarks
            mp_drawing.draw_landmarks(frame, pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        right_index_values = [ pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_INDEX].x, pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_INDEX].y, pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_INDEX].visibility]
        Left_index_values = [pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_INDEX].x , pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_INDEX].y , pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_INDEX].visibility]
        NodeValuesX = []
        NodeValuesY = []
        NodeValuesV = []

        
        for i in range(32):  # MediaPipe has 33 pose landmarks
            NodeValuesX.append(pose_landmarks.landmark[i].x)
            NodeValuesY.append(pose_landmarks.landmark[i].y)
            NodeValuesV.append(pose_landmarks.landmark[i].visibility)
            #X.write(str(pose_landmarks.landmark[i].x))
            #X.write(str(pose_landmarks.landmark[i].y))
            #X.write(str(pose_landmarks.landmark[i].visibility))
            i += 1
        cv2.imshow("Frame", frame)
        cv2.waitKey(1)
        #AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
 
    cap.release()
    cv2.destroyAllWindows()

ZtoXTrack()

#KILL THE CAMERA 

