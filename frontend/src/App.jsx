import './App.css'
import VideoResults from './Pages/VideoResults';
import Upload from './Pages/Upload';
import Layout from './components/Layout';
import { createBrowserRouter, RouterProvider } from "react-router-dom";

const router = createBrowserRouter([
    {
        path: "/",
        element: <Layout />,
        children: [
            { path: "/", element: <Upload />},
            { path: "results", element: <VideoResults /> },
        ],
    },
    {
        path: "*",
        element: <div>404 Not Found</div>,
    },
 ])

function App() {
  return (
    <RouterProvider router={router} />
  );
}

export default App
