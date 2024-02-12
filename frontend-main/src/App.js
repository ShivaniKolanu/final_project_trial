import logo from './logo.svg';
import './App.css';

import {BrowserRouter, Routes, Route, Link} from 'react-router-dom';
import LandingPage from "./pages/LandingPage";
import LoginPage from "./pages/LoginPage";
import RegisterPage from "./pages/RegisterPage";
import WelcomePage from "./pages/WelcomePage";
import FaceHomePage from "./pages/FaceHomePage";
import Registration from './pages/Registration';
import Recognition from './pages/Recognition';
import FaceReg from './pages/FaceReg';
import FaceRec from './pages/FaceRec';
import BlockchainApp from './pages/BlockchainApp';
function App() {
  // const navigate = useNavigate();
  // const [provider, setProvider] = useState(null);
  // const [account, setAccount] = useState(null);
  // const [isConnected, setIsConnected] = useState(false);
  // const [votingStatus, setVotingStatus] = useState(true);
  // const [remainingTime, setremainingTime] = useState('');
  // const [candidates, setCandidates] = useState([]);
  // const [number, setNumber] = useState('');
  // const [CanVote, setCanVote] = useState(true);


  // async function connectToMetamask() {
  //   if (window.ethereum) {
  //     try {
  //       const provider = new ethers.providers.Web3Provider(window.ethereum);
  //       setProvider(provider);
  //       await provider.send("eth_requestAccounts", []);
  //       const signer = provider.getSigner();
  //       const address = await signer.getAddress();
  //       setAccount(address);
  //       console.log("Metamask Connected : " + address);
  //       setIsConnected(true);

  //       if(isConnected){
  //         navigate("/blockConnected")
  //       }
  //       console.log("Is Connected:", isConnected);

  //       // canVote();
  //     } catch (err) {
  //       console.error(err);
  //     }
  //   } else {
  //     console.error("Metamask is not detected in the browser")
  //   }
  // }



  return (
    <div className="vh-100 gradient-custom">
    <div className="container">
      <h1 className="page-header text-center">React and Python Flask Login Register</h1>

      <BrowserRouter>
        <Routes>
            <Route path="/" element={<LandingPage />} />
            <Route path="/login" element={<LoginPage />} />    
            <Route path="/register" element={<RegisterPage />} /> 
            <Route path="/welcome" element={<WelcomePage />} /> 
            <Route path="/facehome" element={<FaceHomePage/>} />
            <Route path="/reg" element={<Registration/>} />
            <Route path="/rec" element={<Recognition/>} />
            <Route path="/facereg" element={<FaceReg/>} />
            <Route path="/facerec" element={<FaceRec/>} />
            <Route path="/blockapp" element={<BlockchainApp/>} />

            {/* <Route path="/blockconnected" element={<BlockchainConnected account = {account}/>}/> */}
            {/* {isConnected? <Route path="/blockconnected" element={<BlockchainConnected account = {account}/>}/> : <Route path="/blockmain" element={<BlockchainMain connectWallet={connectToMetamask} />} />} */}
            {/* <Route path = "/blockconnected" element={<BlockchainConnected account/>} /> */}
            {/* <Route path="/blockconnected" element = {isConnected ? (<BlockchainConnected account = {account}/> ): ( <BlockchainMain/>)}/> */}
                               
        </Routes> 
      </BrowserRouter>
    </div>
    </div>
  );
}

export default App;
