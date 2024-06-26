import { createSlice } from "@reduxjs/toolkit"

export const resultsSlice = createSlice({
    name: "results",
    initialState: {
        video: "",
        actions: [],
        time: 0,
        gtValues: [],
        IsGT: false
    },
    reducers: {
        setResults: (state, action) => {
            state.actions = action.payload.action;
            state.time = action.payload.time;
            state.video = action.payload.video;
            state.gtValues = action.payload.gtValues;
            state.IsGT = action.payload.IsGT;
        },
    },
}); 

export const { setResults } = resultsSlice.actions;
export default resultsSlice.reducer;