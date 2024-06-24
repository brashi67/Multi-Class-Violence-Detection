import { useEffect } from "react";
import { useSelector } from "react-redux";

const AllEvents = () => {
    let timeout;

    let actions = useSelector((state) => state.results.actions);
    const time = useSelector((state) => state.results.time);

    useEffect(() => {
        actions = Object.fromEntries(
            Object.entries(actions).filter(([key, value]) => value.length > 0)
        );
    });

    const playVideo = (div) => {
        clearTimeout(timeout);

        const video = document.querySelector("video");
        const time = div.getAttribute("time");
        const endtime = div.getAttribute("endtime");

        video.currentTime = time;
        video.play();
        const timer = (endtime - time) * 1000;

        timeout = setTimeout(() => {
            video.pause();
        }, timer);
    };

    return (
        <div className="flex flex-col w-72 bg-zinc-700 rounded-xl p-3 relative">
            <div className="absolute font-bold top-0 left-0 rounded-t-xl w-full h-10 flex justify-center items-center bg-zinc-600">
                ALL EVENTS
            </div>
            <div className="relative top-11">
                {Object.keys(actions).map((key, index) => {
                    return actions[key].map((action, index) => {
                        return (
                            <div
                                key={index}
                                time={action.start_time}
                                endtime={action.end_time}
                                title={key}
                                onClick={(e) => playVideo(e.target)}
                                className="relative selection h-10 hover:cursor-pointer  bg-zinc-600 border-2 border-zinc-500 rounded-lg px-3 m-1 flex items-center justify-between">
                                <div className=" font-bold text-start pointer-events-none">
                                    {key}
                                </div>
                                <div className="absolute top-0 right-0 pointer-events-none rounded-r-lg  flex items-center justify-center font-bold text-start border-l-2 border-zinc-500 h-full w-24">
                                    {action.start_time.toFixed(0)}s - {action.end_time.toFixed(0)}s
                                </div>
                            </div>
                        );
                    });
                })}
            </div>
        </div>
    );
};

export default AllEvents;
