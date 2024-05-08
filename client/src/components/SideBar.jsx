import './SideBar.css'
import KeyboardArrowLeftIcon from '@mui/icons-material/KeyboardArrowLeft';
import KeyboardArrowRightIcon from '@mui/icons-material/KeyboardArrowRight';
import { useState } from 'react';
import ShowData from './Data';
import FlexBetween from './FlexBetween';
import { Box, Slider, Typography } from '@mui/material';
import LoopIcon from '@mui/icons-material/Loop';
import { useSelector } from 'react-redux';
import { useEffect } from 'react';
import { useDispatch } from 'react-redux';
import { switchModel } from '../state';

const SideBar = () => {
    const [arrowType, setArrowType] = useState('left');
    const [topics, setTopics] = useState(5);
    const [words, setWords] = useState(7);
    const plsa = useSelector((state) => state.result.plsa);
    const data = useSelector((state) => state.result.document);
    const [model, setModel] = useState(data);
    const [modelName, setModelName] = useState('lda');
    // const [Mydocument, setDocument] = useState(data[0]);
    const [MydocumentName, setMyDocumentName] = useState('doc1');
    const [Mydocument, setDocument] = useState(model[0]);
    const dispatch = useDispatch();

    const ShowBar = () => {
        var sidebar = document.getElementById('sidebar');
        if (sidebar.style.right === '0px') {
            sidebar.style.right = '-550px';
            setArrowType('left');
        } else {
            sidebar.style.right = '0';
            setArrowType('right');
        }
    };

    const SwitchDocument = () => {
        if(Mydocument === model[0]){
            setDocument(model[1]);
        }
        else{
            setDocument(model[0]);
        }
    };

    const SwitchModel = () => {
        if(modelName === 'lda'){
            setModel(plsa);
            setModelName('plsa');            
            dispatch(switchModel());
        }
        else{
            setModel(data);
            setModelName('lda');           
            dispatch(switchModel());
        }
    };

    useEffect(() => {
        if(MydocumentName === 'doc1'){
            setDocument(model[0]);
        }
        else{
            setDocument(model[1]);
        }
        
    }, [model]);

    return(
    <div>
        <div 
            id="sidebar"
        >
            {arrowType === 'left' ? <KeyboardArrowLeftIcon 
                onClick={ShowBar}
                className='leftarrow'
                sx={{
                position: "absolute",
                left:'-30px',
                top:"50%",
                width:"30px",
                height:"40px",
            }}/> : <KeyboardArrowRightIcon 
                onClick={ShowBar}
                className='leftarrow'
                sx={{
                position: "absolute",
                left:'-30px',
                top:"50%",
                width:"30px",
                height:"40px",
            }}/>}
            <Box className="controller">
                <FlexBetween padding="0.25rem 1rem">
                    <Typography sx={{
                        fontSize:'12px'
                    }}>
                        Topics Show Number
                    </Typography>
                    <Slider
                        value={topics}
                        min={1}
                        max={5}
                        sx={{
                            flexBasis:"60%",
                            marginLeft:"auto",
                            margin:"0 0.5rem"
                        }}
                        onChange={(event) => setTopics(event.target.value)}
                    />
                    <Typography sx={{flexBasis:"5%"}}>
                        {topics}
                    </Typography>
                </FlexBetween>
                <FlexBetween padding="0.25rem 1rem">
                    <Typography sx={{
                        fontSize:'12px'
                    }}>
                        Words Show Number
                    </Typography>
                    <Slider
                        value={words}
                        min={5}
                        max={20}
                        sx={{
                            flexBasis:"60%",
                            marginLeft:"auto",
                            margin:"0 0.5rem"
                        }}
                        onChange={(event) => setWords(event.target.value)}
                    />
                    <Typography sx={{flexBasis:"5%"}}>
                        {words}
                    </Typography>
                </FlexBetween>
                <FlexBetween 
                    sx={{
                        padding:'0.25rem 9rem'
                }}>
                    <Typography sx={{
                        fontSize:'12px',
                    }}>
                        {modelName === 'lda' ? "Switch Model to PLSA" : "Switch Model to LDA"}
                    </Typography>
                    <LoopIcon className='rotatable' onClick={SwitchModel}/>
                </FlexBetween>
                <FlexBetween 
                    sx={{
                        padding:'0.25rem 9rem'
                }}>
                    <Typography sx={{
                        fontSize:'12px',
                    }}>
                        {Mydocument === model[0] ? "Switch to Document2 Distribution" : "Switch to Document1 Distribution"}
                    </Typography>
                    <LoopIcon className='rotatable' onClick={SwitchDocument}/>
                </FlexBetween>
            </Box>
            
            <div style={{overflowY: "auto",marginLeft:"auto",height:'100%'}}>
                <ShowData document={Mydocument} topics={topics} words={words}/>
            </div>
        </div>
    </div>
    )
};

export default  SideBar;