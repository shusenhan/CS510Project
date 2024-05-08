import { createSlice } from "@reduxjs/toolkit";

const initialState = {
    result:{
        document:[null,null],
        similarity:0,
        summary:"",
        plsa:[null,null,null]
    },
    document1:'',
    document2:'',
    model:'lda'
}

export const Slice = createSlice({
    name: "document",
    initialState,
    reducers:{
        setDocument1: (state, action) => {
            state.document1 = action.payload.document;
        },
        setDocument2: (state, action) => {
            state.document2 = action.payload.document;
        },
        setResult: (state, action) => {
            state.result = action.payload.result;
        },
        clearState:(state, action) => {
            state.result = {
                document:[null,null],
                similarity:0,
                summary:""
            }
        },
        switchModel:(state, action) => {
            state.model = state.model === 'lda' ? 'plsa' : 'lda';
        }
    }
});

export const {setDocument1, setDocument2, setResult, clearState, switchModel } = Slice.actions;
export default Slice.reducer;
