import { Box } from "@mui/material";
import { styled } from "@mui/system";

const WidgetWrapper = styled(Box)(({theme}) => ({
    padding: "0.75rem 3rem 0.75rem 3rem",
    backgroundColor: "#629bb5",
    borderRadius: "0.75rem"
}));

export default WidgetWrapper;