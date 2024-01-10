from io import BytesIO
from flask import Flask, jsonify, request, render_template, send_file
from models import db, User, Faces, tbl_faces
from flask_cors import CORS, cross_origin
import face_recognition
from models import Face
import cv2
import base64
import numpy as np
import os
from PIL import Image
import datetime
import dlib
from sklearn.neighbors import KNeighborsClassifier
import joblib
from skimage import transform
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


app = Flask(__name__)
CORS(app, supports_credentials= True)

# app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Set your desired limit in bytes
# app.config['MAX_FORM_MEMORY_SIZE'] = 16 * 1024 * 1024  # Set your desired limit in bytes

app.config['SECRET_KEY'] = 'cairocoders-ednalan'
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres:KShivani@localhost:5432/react_login'

SQLALCHEMY_TRACK_MODIFICATIONS = False
SQLALCHEMY_ECHO = True

db.init_app(app)


  
with app.app_context():
    db.create_all()

@app.route("/")
def hello_world():
    return "Hello, World!"

@app.route("/signup", methods = ["POST"])
def signup():

    email = request.json["email"]
    password = request.json["password"]

    data = User.query.filter_by(email = email).first()
    if data:
        return {"error": "User already exist"}, 500
    else:
        new_data = User(email = email, password = password)
        db.session.add(new_data)
        db.session.commit()
        return jsonify({
            "id": new_data.id,
            "email": new_data.email
        })
    
@app.route("/login", methods = ["POST"])
def login():

    email = request.json["email"]
    password = request.json["password"]

    data = User.query.filter_by(email = email, password = password).first()
    if data:
        return jsonify({
            "id": data.id,
            "email": data.email
        }), 200
    else:
        return {"error": "User does not exist or wrong password"}, 500
    
# Function to perform face recognition
def recognize(input_face, stored_face):
    input_encoding = face_recognition.face_encodings(input_face)[0]
    stored_encoding = face_recognition.face_encodings(stored_face)[0]
    result = face_recognition.compare_faces([input_encoding], stored_encoding, tolerance=0.6)
    return result[0]

# Route to handle face registration
@app.route('/register_face', methods=['POST'])
def register_face():
    try:
        user_id = request.form['user_id']

        # Open webcam and capture image for registration
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()

        # Convert image to binary data
        _, buffer = cv2.imencode('.jpg', frame)
        face_image = buffer.tobytes()

        # Save face image to PostgreSQL
        new_face = Face(user_id=user_id, face_image=face_image)
        db.session.add(new_face)
        db.session.commit()

        return jsonify({'message': 'Face registered successfully', 'face_id' : user_id})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
# Load the pre-trained Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


# Route to serve the stored image
@app.route('/get_face_image/<int:face_id>', methods=['GET'])
def get_face_image(face_id):
    try:
        face = Face.query.get(face_id)
        if face:
            return send_file(BytesIO(face.face_image), mimetype='image/jpeg')
        else:
            return jsonify({'error': 'Face not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# Route for face recognition
@app.route('/recognize_face', methods=['POST'])
def recognize_face():
    try:
        # print(request.files)
        input_image = request.files['input_image'].read()

        if not input_image:
            return jsonify({'message': 'Empty image received'})
        
        # Print the received image and its length for debugging
        print("Received image:", input_image[:50])  # Print the first 50 characters of the image data
        print("Image length:", len(input_image))

        # Convert the base64 image to a NumPy array
        nparr = np.frombuffer(base64.b64decode(input_image), np.uint8)

        # Additional check for empty image
        if nparr.size == 0:
            return jsonify({'message': 'Empty image data after decoding'})

        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                # Additional check for empty image after decoding
        if img is None:
            return jsonify({'message': 'Failed to decode image'})


        # Convert the image to grayscale for face detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect faces in the image
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        if len(faces) > 0:
            return jsonify({'message': 'Face recognized successfully'})
        else:
            return jsonify({'message': 'Face not recognized'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    #     input_face = request.files['input_face'].read()

    #     # Retrieve all faces from the database
    #     all_faces = Face.query.all()

    #     # Compare input face with each stored face
    #     for face in all_faces:
            
    #         stored_face = face.face_image
            
    #         try:
    #             if recognize(input_face, stored_face):
                    
    #                 return jsonify({'user_id': face.user_id, 'message': 'Face recognized successfully'})
    #         except Exception as recognition_error:
    #             print(f"Recognition Error: {recognition_error}")
            
    #     return jsonify({'message': 'Face not recognized'})
    # except Exception as e:
    #     print(f"Error: {e}")
    #     return jsonify({'error': str(e)}), 500


#####################################################
    
registered_data = {}



@app.route('/register_face', methods=['POST'])
def registerFace():
    name = request.form.get("name")
    photo = request.files['photo']
    

    uploads_folder = os.path.join(os.getcwd(), "uploads")

    if not os.path.exists(uploads_folder):
        os.makedirs(uploads_folder)
    
    photo.save(os.path.join(uploads_folder, f'{datetime.date.today()}_{name}.jpg'))

    registered_data[name] = f"{datetime.date.today()}_{name}.jpg"

    response = {"success":True, "name": name}
    return jsonify(response)

@app.route('/login_face', methods=['POST'])

def loginFace():
    photo = request.files['photo']

    uploads_folder = os.path.join(os.getcwd(), "uploads")

    if not os.path.exists(uploads_folder):
        os.makedirs(uploads_folder)
    
    login_filename = os.path.join(uploads_folder, "login_face.jpg")

    photo.save(login_filename)

    login_image = cv2.imread(login_filename)
    gray_image = cv2.cvtColor(login_image,cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    faces = face_cascade.detectMultiScale(gray_image, scaleFactor =  1.1, minNeighbors = 5, minSize = (30,30))

    if len(faces) == 0:
        response = {"success" : False}
        return jsonify(response)
    
    login_image = face_recognition.load_image_file(login_filename)

    login_face_encodings = face_recognition.face_encodings(login_image)

    for name, fileName in registered_data.items():
        registered_photo = os.path.join(uploads_folder, fileName)
        registered_image = face_recognition.load_image_file()

#########################################################################################
@app.route('/reg', methods=['POST'])
def reg():

    name = request.form['name']
    # image = request.files['image']

    # Create the folder if it doesn't exist
    folder_path = 'uploads'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Save the image to the folder
    image_path = os.path.join(folder_path, f'{name}.png')
    # image.save(image_path)

    data = Faces.query.filter_by(name = name).first()
    if data:
        return {"error": "User already exist"}, 500
    else:
        new_data = Faces(name = name, image_path = image_path)
        db.session.add(new_data)
        db.session.commit()

        return jsonify({'message': 'Face registered successfully'})
    
# API endpoint for face recognition
@app.route('/rec', methods=['POST'])
def rec():
    name = request.form['name']
    image = request.files['image']

    data = Faces.query.filter_by(name = name).first()



    if not data:
        return {"error": "User does not exist"}, 500
    else:

        # Convert the captured image to a format suitable for face_recognition library
        # captured_image = face_recognition.load_image_file(image)
        # captured_face_encoding = face_recognition.face_encodings(captured_image)[0]

        # recognized = False
        # # Access the attributes of the Faces object directly
        # registered_face_encoding = face_recognition.face_encodings(face_recognition.load_image_file(data.image_path))[0]

        # # Compare face encodings
        # results = face_recognition.compare_faces([registered_face_encoding], captured_face_encoding, tolerance=0.5)

        # print(results)

        # if results[0]:
        #     recognized = True
        #shape_predictor_68_face_landmarks
        #dlib
       # Load the pre-trained shape predictor model from Dlib
        # shape_predictor_model_path = "shape_predictor_68_face_landmarks.dat"
        # predictor = dlib.shape_predictor(shape_predictor_model_path)

        # # Load the registered face encoding
        # registered_image = face_recognition.load_image_file(data.image_path)
        # registered_face_encoding = face_recognition.face_encodings(registered_image)[0]

        # # Convert the captured image to a NumPy array
        # captured_image = cv2.imdecode(np.frombuffer(image.read(), np.uint8), -1)
        
        # # Detect face landmarks using Dlib's shape predictor model
        # # face_rect = dlib.rectangle(0, 0, captured_image.shape[1], captured_image.shape[0])
        # face_rects = face_recognition.face_locations(captured_image, model="cnn")  # Using CNN for face detection
        # if not face_rects:
        #     return {"error": "Face not detected in the captured image"}, 500

        # # face_landmarks = predictor(captured_image, face_rect)

        # # if not face_landmarks:
        # #     return {"error": "Face not detected in the captured image"}, 500

        # # Use face landmarks for face encoding
        # captured_face_encoding = face_recognition.face_encodings(captured_image, [face_rects[0]])[0]

        # # Compare face encodings
        # results = face_recognition.compare_faces([registered_face_encoding], captured_face_encoding, tolerance=0.6)

        # recognized = results[0]

        #opencv

        #  # Load the registered face encoding
        # registered_image = face_recognition.load_image_file(data.image_path)
        # registered_face_encoding = face_recognition.face_encodings(registered_image)[0]

        # # Convert the captured image to a NumPy array
        # captured_image = cv2.imdecode(np.frombuffer(image.read(), np.uint8), -1)

        # # Use a pre-trained deep learning model for face detection
        # face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        # faces = face_cascade.detectMultiScale(captured_image, scaleFactor=1.3, minNeighbors=5)

        # if faces.shape[0] == 0:
        #     return {"error": "Face not detected in the captured image"}, 500

        # # Get the first detected face for face encoding
        # (x, y, w, h) = faces[0]
        # face_location = (y, x + w, y + h, x)
        # captured_face_encoding = face_recognition.face_encodings(captured_image, [face_location])[0]

        # # Compare face encodings
        # results = face_recognition.compare_faces([registered_face_encoding], captured_face_encoding)

        # recognized = results[0]

        #opencv only - ok ok 

            # Load the registered image and convert it to grayscale
        # print(data.image_path)
        # registered_image = cv2.imread(data.image_path)
        # registered_gray = cv2.cvtColor(registered_image, cv2.COLOR_BGR2GRAY)

        # # Load the captured image and convert it to grayscale
        # captured_image = cv2.imdecode(np.frombuffer(image.read(), np.uint8), -1)
        # captured_gray = cv2.cvtColor(captured_image, cv2.COLOR_BGR2GRAY)

        # # Use a pre-trained deep learning model for face detection
        # face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # # Detect faces in both images
        # registered_faces = face_cascade.detectMultiScale(registered_gray, scaleFactor=1.3, minNeighbors=5)
        # captured_faces = face_cascade.detectMultiScale(captured_gray, scaleFactor=1.3, minNeighbors=5)

        # if len(captured_faces) == 0:
        #     return {"error": "Face not detected in the captured image"}, 500

        # # Get the first detected face in the captured image
        # (x, y, w, h) = captured_faces[0]
        # captured_face = captured_gray[y:y+h, x:x+w]

        # # Iterate over registered faces and compare
        # recognized = False
        # for (rx, ry, rw, rh) in registered_faces:
        #     registered_face = registered_gray[ry:ry+rh, rx:rx+rw]
            
        #     # Resize the registered face to match the captured face size
        #     registered_face_resized = cv2.resize(registered_face, (w, h))

        #     # Compare the two faces using a simple mean squared error
        #     mse = np.sum((captured_face - registered_face_resized) ** 2)
        #     mse /= float(captured_face.shape[0] * captured_face.shape[1])
        #     print(mse)

        #     if mse < 100:  # Adjust this threshold as needed
        #         recognized = True
        #         break
        # Load the registered image and convert it to grayscale
# Load the registered image and convert it to grayscale
        registered_image = cv2.imread(data.image_path, cv2.IMREAD_GRAYSCALE)

        # Load the captured image and convert it to grayscale
        captured_image = cv2.imdecode(np.frombuffer(image.read(), np.uint8), -1)
        captured_gray = cv2.cvtColor(captured_image, cv2.COLOR_BGR2GRAY)

        # Apply histogram equalization for better contrast
        registered_image = cv2.equalizeHist(registered_image)
        captured_gray = cv2.equalizeHist(captured_gray)

        # Use LBPH face recognizer
        face_recognizer = cv2.face.LBPHFaceRecognizer_create()

        # Train the recognizer with the registered face
        face_recognizer.train([registered_image], np.array([1]))

        # Perform face recognition on the captured face
        label, confidence = face_recognizer.predict(captured_gray)
        print(confidence)

        # Set the threshold to determine recognition
        threshold = 60  # Adjust as needed

        recognized = confidence < threshold

    # Return success or failure based on recognition result
    return jsonify({'message': 'Recognition successful'} if recognized else {'message': 'Recognition failed'})


##### face recognition tutorial 
faces_directory = os.path.join(os.path.dirname(__file__), 'faces')

known_faces = []
known_names = []

def get_known_encodings(name):
    # known_faces = []
    # known_names = []

    # for fileName in os.listdir(faces_directory):
    image = face_recognition.load_image_file(os.path.join(faces_directory, name+'.jpg'))
    enncoding = face_recognition.face_encodings(image, num_jitters=5)[0]
    return enncoding
        # known_faces.append(enncoding)
        # known_names.append(os.path.splitext(fileName)[0])

def total_registered():
    return len(os.listdir(faces_directory))


def identify_person(k_encoding):
    
    video_capture = cv2.VideoCapture(0)
    while True:
        ret, frame = video_capture.read() #on webcam ret => tells if capturing or not
        rgb_frame = frame[:,:,::-1] 
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        # print(k_encoding)


        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces([k_encoding], face_encoding)
            # name = 'Unknown'

            if True in matches:
                matched = True
            else:
                matched = False
                
                # matched_indices = [i for i, match in enumerate(matches) if match]

                

        cv2.imshow('Camera', frame)

        if cv2.waitKey(1) & 0xFF == ord('g'):
            break

    video_capture.release()
    cv2.destroyAllWindows()
    return matched




@app.route('/recFace', methods=['POST'])
def recFace():
    name = request.form['name']
    data = Faces.query.filter_by(name = name).first()
    if not data:
        return {"error": "User does not exist"}, 500

    known_encoding = get_known_encodings(name)
    is_matched = identify_person(known_encoding)
    print(is_matched)
    if is_matched:
        return "Face Recognition Successful", 200
    else:
        return "Face Recognition not Successful", 500



# @app.route('/video_feed', methods = ['GET'])
# def video_feed():
#     identify_person()
    

@app.route('/add_user', methods = ['POST', 'GET'])
def add_user():
    name = request.form['name']

    data = Faces.query.filter_by(name = name).first()
    if data:
        return {"error": "User already exist"}, 500

    if not os.path.isdir(faces_directory):
        os.makedirs(faces_directory)

    video_capture = cv2.VideoCapture(0)

    while True:
        ret, frame = video_capture.read() #on webcam ret => tells if capturing or not
        flipped_frame = cv2.flip(frame, 1)

        text = "Press Q to Capture & Save the Image"
        font = cv2.FONT_HERSHEY_COMPLEX
        font_scale = 0.9
        font_color = (0, 0, 200)
        thickness = 2

        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_x = (frame.shape[1] - text_size[0]) // 2
        text_y = (frame.shape[0] - 450)

        cv2.putText(flipped_frame, text, (text_x, text_y), font, font_scale, font_color, thickness, cv2.LINE_AA)

        cv2.imshow('Camera for register', flipped_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            img_name = name+'.jpg'
            cv2.imwrite(faces_directory+'/'+img_name, flipped_frame)
            image_path = os.path.join(faces_directory, img_name)
            new_data = Faces(name = name, image_path = image_path)
            db.session.add(new_data)
            db.session.commit()
            break
    video_capture.release()
    cv2.destroyAllWindows()
    return jsonify({'message': 'Face registered successfully'})    


def dataCollect(id):
    video = cv2.VideoCapture(0)
    facedetect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    count = 0
    while(True):
        ret, frame = video.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facedetect.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
            count = count+1
            cv2.imwrite(os.path.join('datasets', 'User.' + str(id) + '.' + str(count) + '.jpg'), gray[y:y+h, x:x+w])
            cv2.rectangle(frame, (x,y), (x+w, y+h), (50, 50, 255), 1)
        cv2.imshow("Frame", frame)
        k = cv2.waitKey(1)

        if count > 99:
            break

        if k==ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()
    print("Dataset Collection done!")
    return True

def getImageID(path):
    imagePath = [os.path.join(path, f) for f in os.listdir(path)]
    faces = []
    ids = []
    for imagePaths in imagePath:
        faceImage = Image.open(imagePaths).convert('L')
        faceNP = np.array(faceImage)
        Id = (os.path.split(imagePaths)[-1].split(".")[1])
        Id = int(Id)
        faces.append(faceNP)
        ids.append(Id)
        cv2.imshow("Training", faceNP)
        cv2.waitKey(1)

    return ids, faces

def training():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    path = "datasets"
    IDs, facedata = getImageID(path)
    recognizer.train(facedata, np.array(IDs))
    recognizer.write("Trainer.yml")
    cv2.destroyAllWindows()
    print("Training Completed")
    return True

def recognizer(id):
    video = cv2.VideoCapture(0)

    facedetect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("Trainer.yml")
    total_faces = 0
    correct_faces = 0
    recognized = False
    while(True):
        ret, frame = video.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facedetect.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
            total_faces += 1
            serial, conf = recognizer.predict(gray[y:y+h, x:x+w])
            # print(serial)
            print(conf)
            if conf < 60: 
                print(conf)
                print(f"serial : {serial}")
                print(f"id: {id}")
                correct_faces += 1
                if serial == id:
                    cv2.putText(frame, "", (x,y-40), cv2.FONT_HERSHEY_SIMPLEX, 1 , (50, 50, 255), 2)
                    cv2.rectangle(frame, (x,y), (x+w, y+h), (50, 50, 255), 1)
                    recognized = True
                else:
                    cv2.putText(frame, "Please Try Again", (x,y-40), cv2.FONT_HERSHEY_SIMPLEX, 1 , (50, 50, 255), 2)
                    cv2.rectangle(frame, (x,y), (x+w, y+h), (50, 50, 255), 1)
            else:
                cv2.putText(frame, "Unknown", (x,y-40), cv2.FONT_HERSHEY_SIMPLEX, 1 , (50, 50, 255), 2)
                cv2.rectangle(frame, (x,y), (x+w, y+h), (50, 50, 255), 1)
        
        accuracy_percentage = (correct_faces / total_faces) * 100 if total_faces > 0 else 0
        accuracy_text = f"Accuracy: {accuracy_percentage:.2f}%"
        cv2.putText(frame, accuracy_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


        cv2.imshow("Frame", frame)
        k = cv2.waitKey(1)


        if k==ord('q') or recognized:
            break

    video.release()
    cv2.destroyAllWindows()
    print("Recognition Done!")
    return True








            
       
@app.route('/register_face_cv2', methods = ['POST'])
def register_face_cv2():
    name = request.form['name']
    data = tbl_faces.query.filter_by(name = name).first()
    if data:
        return {"error": "User already exist"}, 500
    else:
        new_data = tbl_faces(name = name)
        db.session.add(new_data)
        db.session.commit()

    data_fetch = tbl_faces.query.filter_by(name=name).with_entities(tbl_faces.id).first()
    if data_fetch:
        user_id = data_fetch[0]
        # Now, user_id contains the value of the id column for the first record with the specified name
    else:
        # Handle the case where no record is found
        return {"error": "No data found"}, 500
    
    is_dataCollect = dataCollect(user_id)
    is_trained = training()
    if is_dataCollect and is_trained:
        return "Face Data Collection and Training Successful", 200
    else:
        return "Face Data Collection and Training not Successful", 500


    
@app.route('/recognizer_face_cv2', methods = ['POST'])
def recognizer_face_cv2():
    name = request.form['name']
    data = tbl_faces.query.filter_by(name = name).first()
    if not data:
        return {"error": "User Does Not exist"}, 500
    else:
        data_fetch = tbl_faces.query.filter_by(name=name).with_entities(tbl_faces.id).first()
        if data_fetch:
            user_id = data_fetch[0]
            # Now, user_id contains the value of the id column for the first record with the specified name
        else:
            # Handle the case where no record is found
            return {"error": "No data found"}, 500
        is_recognized = recognizer(user_id)
        if is_recognized:
            return "Face Recognition Successful", 200
        else:
            return "Face Recognition not Successful", 500






    

    

    

    
    















if __name__ == "__main__":
    app.run(debug = True)