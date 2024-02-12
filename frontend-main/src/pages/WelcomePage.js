import React, { useState, useRef } from "react";
import axios from 'axios';
import { useLocation } from 'react-router-dom';



export default function WelcomePage(){

  const location = useLocation();
  const searchData = new URLSearchParams(location.search);
  const data = JSON.parse(searchData.get('data'));
  console.log(data);

    const [userId, setUserId] = useState('');
    const [message, setMessage] = useState('');
    const [recognizeFile, setRecognizeFile] = useState(null);
    const [capturedImage, setCapturedImage] = useState(null);

    const videoRef = useRef(null);

  const handleRegister = async () => {
    try {
      const formData = new FormData();
      formData.append('user_id', userId);

      // Send request to register face and capture image from webcam
      const registerResponse = await axios.post('http://localhost:5000/register_face', formData);
      setMessage(registerResponse.data.message);
      // Display the captured image
      console.log(registerResponse);
      const capturedImageResponse = await axios.get(`http://localhost:5000/get_face_image/${registerResponse.data.face_id}`, {
        responseType: 'arraybuffer',
      });

      const imageData = new Blob([capturedImageResponse.data], { type: 'image/jpeg' });
      const imageUrl = URL.createObjectURL(imageData);
      setCapturedImage(imageUrl);
    } catch (error) {
      setMessage(`Error: ${error.response.data.error}`);
    }
  };

  const handleRecognize = async () => {
    try {


        // Open webcam and capture image for recognition
       
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        const video = videoRef.current;
        video.srcObject = stream;

        video.onloadedmetadata = () => {
            video.play();

            setTimeout(async () => {
                const canvas = document.createElement('canvas');
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;
                const context = canvas.getContext('2d');
                context.drawImage(video, 0, 0, canvas.width, canvas.height);
        
                // Convert the image to a Blob
                canvas.toBlob(async (imageBlob) => {
                  // Send captured image for recognition
                  const formData = new FormData();
                  formData.append('input_image', imageBlob, 'image.jpg');
        
                  try {
                    const recognizeResponse = await axios.post('http://localhost:5000/recognize_face', formData);
                    setMessage(recognizeResponse.data.message);
                  } catch (error) {
                    setMessage(`Error: ${error}`);
                  }
        
                  // Stop the webcam stream
                  stream.getTracks().forEach((track) => track.stop());
                }, 'image/jpeg');
              }, 2000); // Delay in milliseconds
            };
          } catch (error) {
            setMessage(`Error: ${error}`);
          }
        };

        // Capture image after a delay (adjust as needed)

//             setTimeout(async () => {

//                 const canvas = document.createElement('canvas');
//                 canvas.width = video.videoWidth;
//                 canvas.height = video.videoHeight;
//                 const context = canvas.getContext('2d');
//                 context.drawImage(video, 0, 0, canvas.width, canvas.height);

//                 // const imageDataURL = canvas.toDataURL('image/jpeg');
//                 // Convert the image to a Blob
//                 const imageBlob = await new Promise((resolve) => canvas.toBlob(resolve, 'image/jpeg'));
//                 console.log(imageBlob);

//                 // Send captured image for recognition
//                 const formData = new FormData();
//                 // formData.append('input_image', imageBlob);
//                 formData.append('input_image', new File([new Blob([imageBlob])], 'image.jpeg'));
                

//                 const recognizeResponse = await axios.post('http://localhost:5000/recognize_face', formData, {
//                     headers: {
//                         'Content-Type': 'multipart/form-data',
//                     },
//                 });
                
//                 setMessage(recognizeResponse.data.message);

//                 // Stop the webcam stream
//                 stream.getTracks().forEach((track) => track.stop());
//             }, 5000); // Delay in milliseconds
//         };
//     } catch (error) {
//             setMessage(`Error: ${error}`);
//     }
//   };

    return(
        <div>
           <p>Data from login response: {data.id}</p>
      <h1>Face Recognition System</h1>
      <div>
        <h2>Register Face</h2>
        <label>User ID:</label>
        <input type="text" value={userId} onChange={(e) => setUserId(e.target.value)} />
        <button onClick={handleRegister}>Register</button>
        <div>
          <p>{message}</p>
          {capturedImage && <img src={capturedImage} alt="Captured Face" />}

        </div>
      </div>
      <div>
        <h2>Recognize Face</h2>
        <video ref={videoRef} width="640" height="480" />
        <button onClick={handleRecognize}>Recognize</button>
      </div>
      <div>
        <p>{message}</p>
      </div>
    </div>
    );
}