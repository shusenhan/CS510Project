import Topic from "./Topic";
import BasicTabs from "./TabSelector.tsx";

const Corpus = ({corpus, topicLimit=5, wordLimit=7}) => {
    return(
        <div style={{overflowY: "auto"}}>
            <BasicTabs corpus={corpus} topicLimit={topicLimit} wordLimit={wordLimit}/>
        </div>
    )
};

export default Corpus;