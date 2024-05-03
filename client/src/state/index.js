import { createSlice } from "@reduxjs/toolkit";

const initialState = {
    result:null
}

export const Slice = createSlice({
    name: "document",
    initialState,
    reducers:{
        setResult: (state, action) => {
            state.result = action.payload.result;
        }
    }
});

export const {setResult } = Slice.actions;
export default Slice.reducer;
