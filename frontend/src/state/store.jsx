import { configureStore } from "@reduxjs/toolkit"
import resultReducer from "./resultSlice"

export default configureStore({
    reducer: {
        results: resultReducer,
    }
})