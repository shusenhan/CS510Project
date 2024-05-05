import { 
    Typography,
    Button,
} from "@mui/material";
import FlexBetween from "../../components/FlexBetween";
import './navbar.css';

const NavBar = () => {

    return(
        <FlexBetween padding="0.35rem 6%" backgroundColor="#4B7AB2">
            <FlexBetween gap="2rem">
                <Typography
                    className=".libre-baskerville-bold"
                    fontSize="clamp(1rem, 1.8rem, 2rem)"
                    color="black"
                >
                    FindTopic
                </Typography>
            </FlexBetween>
        </FlexBetween>
    )
}

export default NavBar;