import React, { useState } from 'react';
import Webcam from 'react-webcam';
import axios from 'axios';

export default function Registration(){

      // Convert base64 image to Blob
    function dataURItoBlob(dataURI) {
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
          setImage(dataURItoBlob(imageSrc));
          console.log('success to capture image.');
        } else {
          console.error('Failed to capture image.');
        }
      };
  
    const registerFace = async () => {
        const formData = new FormData();
        formData.append('name', name);
        console.log('Image value:', image);

        formData.append('image', image, 'image.png'); // Use the raw image data as a blob

        const response = await axios.post('http://localhost:5000/reg', formData);
        console.log(response.data);
        // try {
        // const response = await axios.post('http://localhost:5000/reg', formData);
        // console.log(response.data);
        // } catch (error) {
        // console.error('Error registering face:', error.response.data);
        // }

      // Send image data to the Flask server
    //   const response = await axios.post('http://localhost:5000/reg', {
    //     name: name,
    //     image: image,
    //   });
  
    //   // Handle the response (you can redirect or show a message based on success/failure)
    //   console.log(response.data);
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
            <button onClick={registerFace}>Register Face</button>
        </div>
    );



    
}