import cv2
import os
video = cv2.VideoCapture(0)

facedetect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("Trainer.yml")

name_list = ["", "Shivani", "Durga"]
total_faces = 0
correct_faces = 0
while(True):
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        total_faces += 1
        serial, conf = recognizer.predict(gray[y:y+h, x:x+w])
        # print(conf)
        if conf < 50:
            correct_faces += 1
            cv2.putText(frame, name_list[serial], (x,y-40), cv2.FONT_HERSHEY_SIMPLEX, 1 , (50, 50, 255), 2)
            cv2.rectangle(frame, (x,y), (x+w, y+h), (50, 50, 255), 1)
        else:
            cv2.putText(frame, "Unknown", (x,y-40), cv2.FONT_HERSHEY_SIMPLEX, 1 , (50, 50, 255), 2)
            cv2.rectangle(frame, (x,y), (x+w, y+h), (50, 50, 255), 1)
    
    accuracy_percentage = (correct_faces / total_faces) * 100 if total_faces > 0 else 0
    accuracy_text = f"Accuracy: {accuracy_percentage:.2f}%"
    cv2.putText(frame, accuracy_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


    cv2.imshow("Frame", frame)
    k = cv2.waitKey(1)


    if k==ord('q'):
        break

video.release()
cv2.destroyAllWindows()
print("Recognition Done!")