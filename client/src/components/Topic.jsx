import * as React from 'react';
import Table from '@mui/material/Table';
import TableBody from '@mui/material/TableBody';
import TableCell from '@mui/material/TableCell';
import TableContainer from '@mui/material/TableContainer';
import TableHead from '@mui/material/TableHead';
import TableRow from '@mui/material/TableRow';
import Paper from '@mui/material/Paper';

const Topic = ({words, limit=7}) => {

    return(
        <div>
            <div style={{
                padding:"0 10px 10px 10px",
                fontSize:"16px",
                fontWeight:700
            }}>
                Topic Probability: {Number(words['prob'].toPrecision(3))}
            </div>
            <TableContainer component={Paper}>
                <Table sx={{ width: '100%' }} aria-label="simple table">
                    <TableHead>
                        <TableRow>
                            <TableCell 
                                align='center' 
                                sx={{
                                    width:'50%',
                                    color:'black',
                                    fontWeight:700
                            }}>Word</TableCell>
                            <TableCell 
                                align='center' 
                                sx={{
                                    width:'50%',
                                    color:'black',
                                    fontWeight:700
                                }}>Probability</TableCell>
                        </TableRow>
                    </TableHead>

                    <TableBody>
                        {Object.entries(words['words'])
                            .sort((a, b) => b[1] - a[1])
                            .slice(0,limit)
                            .map(([word, prob],index) => {
                                
                                return (
                                <TableRow 
                                    key={word} 
                                    sx={{ '&:last-child td, &:last-child th': { border: 0 }, 
                                    height:'14px',
                                    padding:"10px 7px"
                                }}>
                                    
                                    <TableCell 
                                        align='center' 
                                        sx={{
                                            width:'50%',
                                            fontSize:'14px', 
                                            height:'16px',
                                            padding:"10px 7px",
                                    }}>{word}</TableCell>
                                    <TableCell 
                                        align='center' 
                                        sx={{
                                            width:'50%', 
                                            fontSize:'14px', 
                                            height:'16px',
                                            padding:"10px 7px",
                                    }}>{Number(prob.toPrecision(3))}</TableCell>
                                </TableRow>
                        )})}
                    </TableBody>
                </Table>
            </TableContainer>
        </div>
    )
};

export default Topic;