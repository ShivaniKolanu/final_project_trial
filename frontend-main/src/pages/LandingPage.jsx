import React, { } from "react";
 
import {Link} from 'react-router-dom';
 
import Button from '@mui/material/Button';

export default function LandingPage(){
 
  return (
    <div>
        <div className="container h-100">
            <div className="row h-100">
                <div className="col-12">
                    <h1>Welcome to this React Applicationnn</h1>
                    <p><Link to="/login" className="btn btn-success">Login</Link> | <Link to="/register" className="btn btn-success">register</Link> </p>
                    <Button variant="outlined" href = "/login">Login</Button>
                    <Button variant="outlined" href = "/register">Register</Button>
                </div>
            </div>
        </div>
    </div>
  );
}