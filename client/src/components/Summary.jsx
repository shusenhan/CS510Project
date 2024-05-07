import { useSelector } from "react-redux";
import { Typography } from "@mui/material";
import { useEffect, useState } from "react";
import * as React from 'react';
import Table from '@mui/material/Table';
import TableBody from '@mui/material/TableBody';
import TableCell from '@mui/material/TableCell';
import TableContainer from '@mui/material/TableContainer';
import TableHead from '@mui/material/TableHead';
import TableRow from '@mui/material/TableRow';
import Paper from '@mui/material/Paper'

const Summary = ({topicLimit, wordLimit}) => {
    const {similarity, document, summary, bert_sim, correlation} = useSelector((state) => state.result)
    const [overlappedWordsProb, setOverlappedWordsProb] = useState(null);
  
    function GetTopWordsProb (Adocument) {
        const topicsArray = Object.values(Adocument);

        // 对 topics 按 prob 降序排序
        const sortedTopics = topicsArray.sort((a, b) => b.prob - a.prob);

        // 获取前五个高 prob 的 topics
        const topFiveTopics = sortedTopics.slice(0, topicLimit);

        // 初始化用于存储结果的对象
        let topWordsProb = {};

        // 提取每个 topic 的 top 20 词汇并累加概率
        topFiveTopics.forEach(topic => {
            const wordsEntries = Object.entries(topic.words);
            const sortedWords = wordsEntries.sort((a, b) => b[1] - a[1]).slice(0, wordLimit);

            sortedWords.forEach(([word, prob]) => {
                if (topWordsProb[word]) {
                    topWordsProb[word] += prob;
                } else {
                    topWordsProb[word] = prob;
                }
            });
        });
        return topWordsProb;
    };

    function GetOverlappedWordsProb ()  {
        const D1WP = GetTopWordsProb(document[0]);
        const D2WP = GetTopWordsProb(document[1]);

        // console.log("D1WP:",D1WP)
        // console.log("D2WP:",D2WP)

        const commonWords = {}
        for (const word in D1WP) {
            if (word in D2WP) {
                commonWords[word] = [D1WP[word], D2WP[word]];
            }
        }

        setOverlappedWordsProb(commonWords);
    }

    useEffect(() => {
        if(document[0] !== null){
            GetOverlappedWordsProb();
        }
    }, [topicLimit, wordLimit]);

    if(!document[0]){
        return(
            <div> loading data.....</div>
        )
    }

    return(
        <div
            style={{
                borderRadius:'4px',
                backgroundColor:"white",
                boxShadow: "0 0 5px rgba(131, 131, 131, 0.3)",
        }}>
            <Typography 
                component="div" 
                sx={{
                    padding:"12px 12px",
                    fontSize:"20px",
                    color:"black",
                    fontWeight:700,
            }}>
                Summary
            </Typography>
            <Typography sx={{padding:"5px 5px 2px 5px", fontSize:'13px', fontWeight:700}} component="div">
                Cosine Similarity: { similarity !== null && Number(similarity.toPrecision(3))}
            </Typography>
            <Typography sx={{padding:"2px 5px 2px 5px", fontSize:'13px', fontWeight:700}} component="div">
                Bert Similarity: { bert_sim !== null && Number(bert_sim.toPrecision(3))}
            </Typography>
            <Typography sx={{padding:"2px 5px 2px 5px", fontSize:'13px', fontWeight:700}} component="div">
                Pearson Correlation: { correlation !== null && Number(correlation.toPrecision(3))}
            </Typography>
            <Typography sx={{padding:"5px 5px 15px 5px", fontSize:'14px'}} component="div">
                Possible Topic: {summary}
            </Typography>
            <Typography component="div" sx={{padding:"10px 5px", fontSize:'14px'}}>
                Common Words in High Probability Topic
            </Typography>
            <div>
                <TableContainer component={Paper}>
                    <Table sx={{ width: '100%' }}>
                        <TableHead>
                            <TableRow>
                                <TableCell align='center' sx={{width:"33%", fontWeight:700}}>Word</TableCell>
                                <TableCell align='center' sx={{width:"33%", fontWeight:700}}>Prob in D1</TableCell>
                                <TableCell align='center' sx={{width:"33%", fontWeight:700}}>Prob in D2</TableCell>
                            </TableRow>
                        </TableHead>

                        {overlappedWordsProb && document && (<TableBody>
                            {Object.entries(overlappedWordsProb)
                                .map(([word, prob]) => (
                                    <TableRow key={word} sx={{ '&:last-child td, &:last-child th': { border: 0 } }}>
                                        <TableCell align='center' sx={{width:"33%", fontSize:'12px'}}>{word}</TableCell>
                                        <TableCell align='center' sx={{width:"33%", fontSize:'12px'}}>{Number(prob[0].toPrecision(3))}</TableCell>
                                        <TableCell align='center' sx={{width:"33%", fontSize:'12px'}}>{Number(prob[1].toPrecision(3))}</TableCell>
                                    </TableRow>
                            ))}
                        </TableBody>)}
                    </Table>
                </TableContainer>
            </div>
        </div>
    )
};

export default Summary;