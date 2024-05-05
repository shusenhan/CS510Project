import * as React from 'react';
import Tabs from '@mui/material/Tabs';
import Tab from '@mui/material/Tab';
import Typography from '@mui/material/Typography';
import Box from '@mui/material/Box';
import Topic from './Topic';
import Summary from './Summary';

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

interface TopicDistribute {
  prob: number;
  words: { [key: string]: number };
}

function CustomTabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props;

  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`simple-tabpanel-${index}`}
      aria-labelledby={`simple-tab-${index}`}
      {...other}
    >
      {value === index && (
        <Box sx={{ p: 3 }}>
          <Typography component="div">{children}</Typography>
        </Box>
      )}
    </div>
  );
}

function a11yProps(index: number) {
  return {
    id: `simple-tab-${index}`,
    'aria-controls': `simple-tabpanel-${index}`,
  };
}

export default function BasicTabs({corpus, topicLimit=5, wordLimit=7}) {
  const [value, setValue] = React.useState(0);

  const handleChange = (event: React.SyntheticEvent, newValue: number) => {
    setValue(newValue);
  };

  return (
    <Box sx={{ width: '100%' }}>
        <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
            <Tabs 
                value={value} 
                onChange={handleChange} 
                variant="scrollable"
                sx={{
                    height:"25px", 
                    minHeight:'25px',
                    margin:"0 10px"
            }}>
                <Tab 
                    sx={{
                        fontSize:'12px', 
                        color:'black',
                        height:"25px", 
                        minHeight:'25px',
                        borderRadius:'7px'
                    }} 
                    label='Summary'
                    {...a11yProps(0)}
                />
            {Object.entries(corpus).slice(0,topicLimit).map(([topicIndex]) => (
                    <Tab 
                      key={topicIndex}
                      sx={{
                          fontSize:'12px', 
                          color:'black',
                          height:"25px", 
                          minHeight:'25px',
                          borderRadius:'7px'
                      }} 
                      label={'Topic ' + (Number(topicIndex) + 1)} 
                      {...a11yProps(Number(topicIndex)+1)}
                    />))}
            </Tabs>
        </Box>

        
        
        {Object.entries(corpus as{[key: string]: TopicDistribute})
          .sort((a,b) => b[1]['prob'] - a[1]['prob'])
          .slice(0,topicLimit)
          .map(([topicIndex, words], index) => (
            <CustomTabPanel key={topicIndex} value={value} index={index+1}>
                <Topic words={words} limit={wordLimit}/>
            </CustomTabPanel>
        ))}

        <CustomTabPanel value={value} index={0}>
          <Summary topicLimit={topicLimit} wordLimit={wordLimit}/>
        </CustomTabPanel>
    </Box>
  );
}