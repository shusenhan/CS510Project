import { createSlice } from "@reduxjs/toolkit";

const initialState = {
    result:{
        document:[null,null],
        similarity:0,
        summary:""
    },
    document1:'',
    document2:'',
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
        }
    }
});

export const {setDocument1, setDocument2, setResult, clearState } = Slice.actions;
export default Slice.reducer;
