import * as React from 'react';
import { useState } from 'react';
import Button from '@mui/material/Button';
import InputAdornment from '@mui/material/InputAdornment';
import axios from 'axios';
import TextField from '@mui/material/TextField';
import AccountCircle from '@mui/icons-material/AccountCircle';
import {useNavigate} from "react-router-dom";


export default function FaceRec(){
    const navigate = useNavigate();

    const [name, setName] = useState('');
    const recognizeFace = async () => {
        const formdata = new FormData();
        formdata.append('name', name);

        console.log('Name is: ', name);
        const response = await axios.post('http://localhost:5000/recFace', formdata);
        console.log(response.data);
    };
    const recognizeFace_cv2 = async () => {
        const formdata = new FormData();
        formdata.append('name', name);

        console.log('Name is: ', name);
        const response = await axios.post('http://localhost:5000/recognizer_face_dl', formdata);
        
        console.log(response.data);
        if(response.status === 200) {
            navigate("/welcome");
        } else if(response.status === 500){
            alert("Not Working!");
        }
    };

    return(
        <div>
            <h2>Recognizing Face</h2>
                <p>
                <TextField
                    id="input-with-icon-textfield"
                    value={name}
                    onChange={(e) => setName(e.target.value)}
                    label="Name for Recognition"
                    InputProps={{
                    startAdornment: (
                        <InputAdornment position="start">
                        <AccountCircle />
                        </InputAdornment>
                    ),
                    }}
                    variant="standard"
                />
            </p>

            <p>
                <Button onClick={recognizeFace_cv2} variant="contained">Submit</Button>
            </p>
        </div>
    );



}