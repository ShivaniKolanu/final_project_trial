import cv2
import os
import face_recognition
import datetime

from flask import app
from models import db, Faces

# Get the absolute path to the 'faces' directory within the 'backend' folder
faces_directory = os.path.join(os.path.dirname(__file__), 'faces')

#load known faces and their names from the 'faces' folder.

known_faces = []
known_names = []

for fileName in os.listdir(faces_directory):
    image = face_recognition.load_image_file(os.path.join(faces_directory, fileName))
    enncoding = face_recognition.face_encodings(image, num_jitters=5)[0]
    known_faces.append(enncoding)
    known_names.append(os.path.splitext(fileName)[0])


video_capture = cv2.VideoCapture(0)

# print(known_names)

while True:
    ret, frame = video_capture.read() #on webcam ret => tells if capturing or not
    rgb_frame = frame[:,:,::-1] 
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    recognized_names = []
    confidences = []

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_faces, face_encoding)
        name = 'Unknown'
        # distances = face_recognition.face_distance(known_faces, face_encoding)
        # min_distance_index = distances.argmin()
        # min_distance = distances[min_distance_index]
        # name = known_names[min_distance_index]
        # if min_distance < 0.4:  # You can adjust this threshold based on your needs
        #     print(f'min is {min_distance}')
        #     # recognized_names.append(name)
        #     confidences.append(1 - min_distance)  # Confidence is the inverse of distance


        if True in matches:
            matched_indices = [i for i, match in enumerate(matches) if match]
            for index in matched_indices:
                name = known_names[index]
                recognized_names.append(name)
    
    if recognized_names:
        print(f"Detected: {recognized_names}")
        break  # Break the loop if a face is detected
    # print(confidences)
    cv2.imshow('Camera', frame)

    if cv2.waitKey(1) & 0xFF == ord('g'):
        break

video_capture.release()
cv2.destroyAllWindows()


