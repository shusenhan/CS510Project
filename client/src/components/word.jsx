const Word = ({text, prob}) => {

    return(
        <div>
            text:{text} prob:{Number(prob.toPrecision(3))}
        </div>
    )
};

export default Word;