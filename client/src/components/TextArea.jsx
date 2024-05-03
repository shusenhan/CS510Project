import { Box, InputBase, Divider } from "@mui/material";
import { useState, useRef } from "react";
import UploadIcon from '@mui/icons-material/Upload';
import DownloadIcon from '@mui/icons-material/Download';
import DeleteIcon from '@mui/icons-material/Delete';
import "./TextArea.css"

const TextArea = ({textContent, setText}) => {
    const [fileName, setFileName] = useState('FileName');
    const fileInputRef = useRef(null);

    const handleFileChange = (event) => {
        const file = event.target.files[0];

        if (file) {
            setFileName(file.name.replace(".txt", ""));
            const reader = new FileReader();
            reader.readAsText(file); // 读取文件为文本

            reader.onload = (e) => {
                const text = e.target.result;
                setText(text);
            };
        }
    };

    const Upload = () => {
        fileInputRef.current.click(); // 触发input元素的点击事件
    };

    const Delete = () => {
        setText('');
        setFileName('');
    };

    const handleTextChange = (event) =>{
        setText(event.target.value);
    };

    const handleDownload = () => {
        const blob = new Blob([textContent], {type: 'text/plain'});
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = fileName;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(url);
    };

    const handleFileNameChange = (event) => {
        setFileName(event.target.value);
    }

    return(
        <Box sx={{
            width:"100%",
            borderRadius:"10px",
            backgroundColor:"#F4F3F3",
            margin:"0.6rem 0.6rem",
            padding:"0.6rem 0.6rem"
        }}>
            <Box sx={{
                display:"flex", 
                gap:"8px", 
                justifyContent:"flex-start",
                padding:"0 1rem"
            }}>
                <Box flexBasis="70%" >
                    <InputBase
                        value={fileName}
                        onChange={handleFileNameChange}
                        fullWidth/>
                </Box>
                <DeleteIcon className="deleteicon" onClick={Delete} sx={{marginLeft:"auto"}}/>
                <UploadIcon className="icon" onClick={Upload}/>
                <input 
                    style={{display:"none"}} 
                    type="file" 
                    onChange={handleFileChange} 
                    accept=".txt" 
                    ref={fileInputRef}
                />
                <DownloadIcon onClick={handleDownload} className="icon"/>
            </Box>

            <Divider/>

            <textarea 
                className="textarea" 
                value={textContent}
                onChange={handleTextChange}
                style={{
                    backgroundColor:"#E8E7E7",
            }}>
            </textarea>
        </Box>
    )
};

export default TextArea;