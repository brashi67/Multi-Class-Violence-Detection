// import video from "../assets/mewo.mp4";
import Timeline from "../components/Timeline.jsx";
import AllEvents from "../components/AllEvents.jsx";
import { useSelector } from "react-redux";

const VideoResults = () => {
    const video = useSelector((state) => state.results.video);
  return (
      <div>
          <div className="flex relative z-10 gap-2">
              <div className="flex flex-col gap-2">
                  <video width={750} height={500} controls className="rounded-xl">
                      <source src={video} type="video/mp4" />
                  </video>
                  <Timeline />
              </div>
              <AllEvents />
          </div>
      </div>
  );
}

export default VideoResults