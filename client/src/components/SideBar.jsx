import './SideBar.css'
import KeyboardArrowLeftIcon from '@mui/icons-material/KeyboardArrowLeft';
import KeyboardArrowRightIcon from '@mui/icons-material/KeyboardArrowRight';
import { useState } from 'react';
import ShowData from './Data';
import FlexBetween from './FlexBetween';
import { Box, Slider, Typography } from '@mui/material';
import LoopIcon from '@mui/icons-material/Loop';
import { useSelector } from 'react-redux';

const SideBar = () => {
    const [arrowType, setArrowType] = useState('left');
    const [topics, setTopics] = useState(5);
    const [words, setWords] = useState(7);
    const data = useSelector((state) => state.result.document);
    const [Mydocument, setDocument] = useState(data[0]);

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
        if(Mydocument === data[0]){
            setDocument(data[1]);
        }
        else{
            setDocument(data[0]);
        }
    };

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
                        padding:'0.25rem 12rem'
                }}>
                    <Typography sx={{
                        fontSize:'12px'
                    }}>
                        Switch Document
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