import React from "react";
import Button from '@mui/material/Button';
import TextField from '@mui/material/TextField';


const BlockchainConnected = (props) => {

    return (

        <div>
            <h1>Connected Metamask!</h1>
            <p>Metamask Account connected to: {props.account}</p>
            <div>
                <TextField
            id="outlined-number"
            label="Number"
            type="number"
            value={props.number}
            onChange={props.handleNumberChange}
            InputLabelProps={{
                shrink: true,
            }}
            />
            <Button variant="contained" onClick = {props.voteFunction}>Vote!</Button>

            </div>

            <table id="myTable" className="candidates-table">
                <thead>
                <tr>
                    <th>Index</th>
                    <th>Candidate name</th>
                    <th>Candidate votes</th>
                </tr>
                </thead>
                <tbody>
                {props.candidates.map((candidate, index) => (
                    <tr key={index}>
                    <td>{candidate.index}</td>
                    <td>{candidate.name}</td>
                    <td>{candidate.voteCount}</td>
                    </tr>
                ))}
                </tbody>
            </table>
        </div>

    );

    
};



export default BlockchainConnected;