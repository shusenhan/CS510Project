import { 
    Typography,
    Box,
    Button,
} from "@mui/material";
import FlexBetween from "../../components/FlexBetween";
import './navbar.css';

const NavBar = () => {


    return(
        <FlexBetween padding="0.35rem 6%" backgroundColor="#244b73">
            <FlexBetween gap="2rem">
                <Typography
                    className="libre-baskerville-regular-italic"
                    fontSize="clamp(1rem, 1.8rem, 2rem)"
                    color="black"
                    sx={{
                        "&:hover":{
                            color: "#95c8fc",
                            cursor: "pointer"
                        }
                    }}
                >
                    FindTopic
                </Typography>
            </FlexBetween>
            <FlexBetween gap="2rem" padding="0 1rem">
                <Button 
                    variant="contained" 
                    sx={{
                        backgroundColor:"#2884e0",
                        fontSize:"1rem", 
                        padding:"0.15rem 0.75rem 0.15rem 0.75rem"
                }}>
                    START
                </Button>
            </FlexBetween>
        </FlexBetween>
    )
}

export default NavBar;