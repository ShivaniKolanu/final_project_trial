
import { useState, useEffect } from "react";
import {ethers} from "ethers";
import { contractAbi, contractAddress } from "../constant/constant";
import React from "react";
import Button from '@mui/material/Button';


const BlockchainMain = (props) => {

    return (

        <div>
            <h1>Welcome</h1>
            <Button variant="contained" onClick = {props.connectWallet}>Vote Now!</Button>
        </div>

    );

    
};


export default BlockchainMain;