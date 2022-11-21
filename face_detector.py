import cv2
from random import randrange

#Load pretrained data on face frontals from open cv
trained_face_data = cv2.CascadeClassifier('haardcascade_frontal_default.xml')

#Choose an image to detect faces in
#img = cv2.imread('rdj.png')

#Open Default Video Capture
webcam = cv2.VideoCapture(0)

#iterate forever over frames
while True:
    #Read current frame
    successful_frame_read, frame = webcam.read()

    #Change to greyscale
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #Detect Faces
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

    for x, y, w, h in face_coordinates:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow('Face Detector', frame)
    key = cv2.waitKey(1)

    #Press q to quit from ascii value
    if key == 81 or key == 113:
        break

webcam.relase() 


print('Code Completed')