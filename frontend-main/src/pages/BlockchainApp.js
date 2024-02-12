
import { useState, useEffect } from "react";
import {ethers} from "ethers";
import { contractAbi, contractAddress } from "../constant/constant";
import {useNavigate} from "react-router-dom";
import BlockchainMain from "./BlockchainMain";
import BlockchainConnected from "./BlockchainConnected";

export default function BlockchainApp(){
    const navigate = useNavigate();
    const [provider, setProvider] = useState(null);
    const [account, setAccount] = useState(null);
    const [isConnected, setIsConnected] = useState(false);
    const [votingStatus, setVotingStatus] = useState(true);
    const [remainingTime, setremainingTime] = useState('');
    const [candidates, setCandidates] = useState([]);
    const [number, setNumber] = useState('');
    const [CanVote, setCanVote] = useState(true);

    useEffect( () => {
        getCandidates();
        getCurrentStatus();

    }, []);

    async function vote() {
        const provider = new ethers.providers.Web3Provider(window.ethereum);
        await provider.send("eth_requestAccounts", []);
        const signer = provider.getSigner();
        const contractInstance = new ethers.Contract (
          contractAddress, contractAbi, signer
        );
  
        const tx = await contractInstance.vote(number);
        await tx.wait();
        // canVote();
    }

    async function getCandidates() {
        const provider = new ethers.providers.Web3Provider(window.ethereum);
        await provider.send("eth_requestAccounts", []);
        const signer = provider.getSigner();
        const contractInstance = new ethers.Contract (
          contractAddress, contractAbi, signer
        );
        const candidatesList = await contractInstance.getAllVotesOfCandiates();
        console.log(candidatesList);
        const formattedCandidates = candidatesList.map((candidate, index) => {
          return {
            index: index,
            name: candidate.name,
            voteCount: candidate.voteCount.toNumber()
          }
        });
        setCandidates(formattedCandidates);
    }

    async function getCurrentStatus() {
        const provider = new ethers.providers.Web3Provider(window.ethereum);
        await provider.send("eth_requestAccounts", []);
        const signer = provider.getSigner();
        const contractInstance = new ethers.Contract (
          contractAddress, contractAbi, signer
        );
        const status = await contractInstance.getVotingStatus();
        console.log(status);
        setVotingStatus(status);
    }

    async function connectToMetamask() {
        if (window.ethereum) {
          try {
            const provider = new ethers.providers.Web3Provider(window.ethereum);
            setProvider(provider);
            await provider.send("eth_requestAccounts", []);
            const signer = provider.getSigner();
            const address = await signer.getAddress();
            setAccount(address);
            console.log("Metamask Connected : " + address);
            setIsConnected(true);
            console.log("Is Connected:", isConnected);
    
            // canVote();
          } catch (err) {
            console.error(err);
          }
        } else {
          console.error("Metamask is not detected in the browser")
        }
      }
      async function handleNumberChange(e) {
        setNumber(e.target.value);
      }
    
    return(
        <div>
            {/* <BlockchainMain connectWallet = {connectToMetamask}/> */}
            {isConnected ? (<BlockchainConnected
                                 account = {account}
                                 candidates = {candidates}
                                 number = {number}
                                 handleNumberChange = {handleNumberChange}
                                 voteFunction = {vote}/>) : (<BlockchainMain connectWallet = {connectToMetamask}/>)}
        </div>
    );
}