import { Box } from "@mui/material";
import { useSelector } from "react-redux";
import Corpus from "./Corpus";

const ShowData = ({document, topics, words}) => {
    // const data = useSelector((state) => state.result);

    return(
        <Box sx={{overflowY: "auto"}}>
            <Corpus corpus={document} topicLimit={topics} wordLimit={words}/>
        </Box>
    )
};

export default ShowData;