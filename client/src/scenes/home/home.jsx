import NavBar from "../navbar/navbar";
import { Box, Button } from "@mui/material";
import { useState } from "react";
import TextArea from "../../components/TextArea";
import { setResult } from "../../state";
import { useDispatch } from "react-redux";

const HomePage = () => {
    const [text1, setText1] = useState('Please Input Some Contexts or Upload File...');
    const [text2, setText2] = useState('Please Input Some Contexts or Upload File...');
    const dispatch = useDispatch();

    const Compare = async () => {
        const response = await fetch(
            "http://localhost:8000/core/",
            {
                method: "POST",
                headers: {"Content-Type" : "application/json"},
                body: JSON.stringify({
                    text1: text1,
                    text2: text2
                })
            }
        )

        const result = await response.json();
        console.log(result);

        // if(result){
        //     dispatch(
        //         setResult({
        //             result:result
        //         })
        //     );
        // }
    }

    return(
        <Box flex="1" sx={{
            backgroundColor:"#F7DC6F"
        }}>
            <NavBar/>
            <Box
                padding="2rem 6%"
                display="flex"
                gap="1.5rem"
                justifyContent="space-between"
            >
                <TextArea flexBasis="40%" textContent={text1} setText={setText1}/>
                <TextArea flexBasis="40%" textContent={text2} setText={setText2}/>
            </Box>
            <Button sx={{backgroundColor:"green", color:"black"}} onClick={Compare}>
                Compare
            </Button>
        </Box>
    )
};

export default HomePage;