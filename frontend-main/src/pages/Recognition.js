import React, { useState } from 'react';
import Webcam from 'react-webcam';
import axios from 'axios';


export default function Recognition(){
    // Convert base64 image to Blob
    function dataURItoBlob(dataURI) {
        if (!dataURI) return new Blob();  // Return an empty Blob if dataURI is null
    
        const byteString = atob(dataURI.split(',')[1]);
        const arrayBuffer = new ArrayBuffer(byteString.length);
        const uint8Array = new Uint8Array(arrayBuffer);
    
        for (let i = 0; i < byteString.length; i++) {
        uint8Array[i] = byteString.charCodeAt(i);
        }
    
        return new Blob([arrayBuffer], { type: 'image/png' });
    }

    const [name, setName] = useState('');
    const [image, setImage] = useState(null);
    const webcamRef = React.useRef(null);

    const capture = () => {
        const imageSrc = webcamRef.current.getScreenshot();
        if (imageSrc) {
          setImage(imageSrc);
        } else {
          console.error('Failed to capture image.');
        }
      };

    const recognizeFace = async () => {
        const formData = new FormData();
        formData.append('name', name);
        // Convert the base64 image data to a Blob
        const blob = await (await fetch(image)).blob();
          // Log the value of image to the console
        console.log('Image value:', image);
        formData.append('image', dataURItoBlob(image), 'image.png');
    
        // Send the FormData to the Flask server
        const response = await axios.post('http://localhost:5000/rec', formData);
    
        // Handle the response (you can redirect or show a message based on success/failure)
        console.log(response.data);
    };

    return(
        <div>
            <input
            type="text"
            placeholder="Enter your name"
            value={name}
            onChange={(e) => setName(e.target.value)}
            />
            <Webcam ref={webcamRef} />
            <button onClick={capture}>Capture</button>
            <button onClick={recognizeFace}>Recognize Face</button>
        </div>
    );
}