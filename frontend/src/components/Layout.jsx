import { Outlet } from "react-router-dom";
import Navbar from "./Navbar";
import leftDesign from "../assets/left.png";
import rightDesign from "../assets/right.png";

const Layout = () => {
  return (
      <div>
          <Navbar />
          <div
              className="h-full w-96 top-0 left-0 absolute z-0"
              style={{
                  background: `url(${leftDesign})`,
                  backgroundPosition: "left",
                  backgroundRepeat: "no-repeat",
              }}></div>
          <main>
              <Outlet /> {/* Nested routes render here */}
          </main>
          <div
              className="h-full w-96 top-0 right-0 absolute z-0"
              style={{
                  background: `url(${rightDesign})`,
                  backgroundPosition: "right",
                  backgroundRepeat: "no-repeat",
              }}></div>
      </div>
  );
}

export default Layout