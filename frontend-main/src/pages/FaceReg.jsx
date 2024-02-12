import * as React from 'react';
import { useState } from 'react';
import Button from '@mui/material/Button';
import InputAdornment from '@mui/material/InputAdornment';
import axios from 'axios';
import TextField from '@mui/material/TextField';
import AccountCircle from '@mui/icons-material/AccountCircle';


export default function FaceReg(){
    const [name, setName] = useState('');
    const registerFace = async () => {
        const formdata = new FormData();
        formdata.append('name', name);

        console.log('Name is: ', name);
        const response = await axios.post('http://localhost:5000/add_user', formdata);
        console.log(response.data);
    };
    const registerFace_opencv = async () => {
        const formdata = new FormData();
        formdata.append('name', name);

        console.log('Name is: ', name);
        const response = await axios.post('http://localhost:5000/register_face_dl', formdata);
        console.log(response.data);
    };


    return(

        <div>
                <p>
                <TextField
                    id="input-with-icon-textfield"
                    value={name}
                    onChange={(e) => setName(e.target.value)}
                    label="Name for Registration"
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
                <Button onClick={registerFace_opencv} variant="contained">Submit</Button>
            </p>
        </div>
  
    );

}