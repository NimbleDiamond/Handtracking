import cv2
import mediapipe as mp 


#Initialize Mediapipe Hand Solution
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=2,
                       min_detection_confidence=0.7,
                       min_tracking_confidence=0.7)

mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

#Error check fo open camera
if not cap.isOpened():
    print("Error")
    exit()

#MainLoop

while True:

    #Frame by Frame Capture
    success, frame = cap.read()
    if not success:
        break
    
    #FlipFrame
    frame = cv2.flip(frame,1)


    #Convert Colourspace

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    #Process the RGB frame with Mediapipe Hands
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for Hand_landmarks in result.multi_hand_landmarks:
            #Draw Landmarks
            mp_drawing.draw_landmarks(frame, Hand_landmarks, mp_hands.HAND_CONNECTIONS)

    #Display result
    cv2.imshow("Frame", frame)
    cv2.waitKey(1)


cap.release()
cv2.destroyAllWindows()
