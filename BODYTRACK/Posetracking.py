import cv2
import mediapipe as mp
from PosetrackingZY import ZtoXTrack
from mediapipe.framework.formats import landmark_pb2 

# Initialize Mediapipe Hand solution
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands

# I for loop
i = 0

#Does tracky stuff
pose = mp_pose.Pose(static_image_mode=False,
                       min_detection_confidence=0.7,
                       min_tracking_confidence=0.7)

hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=2,
                       min_detection_confidence=0.7,
                       min_tracking_confidence=0.7)


mp_drawing = mp.solutions.drawing_utils

#open the camera
cap = cv2.VideoCapture(0)


# error check to make sure the camera is open
if not cap.isOpened():
    print("Error")
    exit()


#Main loop
while True:
    TrueX = []
    TrueZ = []
    TrueY = []
    TrueV = []
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
    Handresult = hands.process(rgb_frame)

    ONodeValuesX = []
    ONodeValuesY = []
    ONodeValuesV = []

    NodeValuesX, NodeValuesY, NodeValuesV = ZtoXTrack()
    for i in range(33):  # MediaPipe has 33 pose landmarks
        ONodeValuesX.append(pose_landmarks.landmark[i].x)
        ONodeValuesY.append(pose_landmarks.landmark[i].y)
        ONodeValuesV.append(pose_landmarks.landmark[i].visibility)
        i += 1
    for o in range(33):
        TrueY.append((NodeValuesY[o] + ONodeValuesY[o])/2)
        TrueX.append(ONodeValuesX[o])
        TrueZ.append(NodeValuesX[o])
        TrueV.append((ONodeValuesV[o]+ NodeValuesV[o])/2)
        true_landmarks = []
    for i in range(33):
        landmark = landmark_pb2.NormalizedLandmark(
            x=TrueX[i],
            y=TrueY[i],
            z=TrueZ[i] if i < len(TrueZ) else 0,
            visibility=TrueV[i]
        )
        true_landmarks.append(landmark)
    
    true_landmark_list = landmark_pb2.NormalizedLandmarkList(landmark=true_landmarks)
    mp_drawing.draw_landmarks(
        frame,
        true_landmark_list,
        mp_pose.POSE_CONNECTIONS
)
    
    if Handresult.multi_hand_landmarks:
        for Hand_landmarks in Handresult.multi_hand_landmarks:
            #Draw Landmarks
            mp_drawing.draw_landmarks(frame, Hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Display the resulting frame
    cv2.imshow("Frame", frame)
    cv2.waitKey(1)

#KILL THE CAMERA
cap.release()
cv2.destroyAllWindows()
