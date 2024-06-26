import { useEffect } from "react";
import { useSelector } from "react-redux";

const Timeline = () => {
    let actions = useSelector((state) => state.results.actions);
    const time = useSelector((state) => state.results.time).toFixed(0);

    const isGT = useSelector((state) => state.results.IsGT);
    const gtValues = useSelector((state) => state.results.gtValues);

    console.log(actions);
    const size = gtValues.length;
    const predsTime = gtValues.map((pred, index) => {
        return {
            pred: pred,
            time: ((index * 16) / 24).toFixed(0),
            progress: ((index / gtValues.length) * 100).toFixed(2),
        };
    });

    const colorFunc = (pred) => {
        if (pred !== 0) return "yellow";
    };
    
    useEffect(() => {
        const video = document.querySelector("video");
        video.addEventListener("timeupdate", () => {
            const pointer = document.querySelector(".Pointer");
            const videoTime = video.currentTime;
            const progress = (videoTime / time) * 100;
            pointer.style.left = `${progress}%`;
        });
        actions = Object.fromEntries(
            Object.entries(actions).filter(([key, value]) => value.length > 0)
        );
    });

    const playVideo = (div) => { 
        const video = document.querySelector("video");
        const time = div.getAttribute("time");
        const endtime = div.getAttribute("endtime");

        video.currentTime = time;
        video.play();
        const timer = (endtime - time) * 1000;

        setTimeout(() => {
            video.pause();
        }, timer);
    }

    // const actions = {
    //     fighting: [],
    //     shooting: [
    //         { start_time: 0.0, end_time: 12.0 },
    //         { start_time: 17.33, end_time: 23.33 },
    //     ],
    //     explosion: [],
    //     riot: [],
    //     abuse: [],
    //     accident: [],
    // };
    // const time = 60;


    return (
        <>
            <div className="h-16 w-full gap-2 flex flex-col justify-center items-center bg-zinc-700 rounded-xl relative py-3">
                <div className="Pointer absolute font-bold top-0 left-0 rounded w-1 h-full z-20 flex justify-center items-center bg-zinc-200"></div>
                {isGT ? (
                    <div className="h-2/4 w-full relative bg-slate-500">
                        {predsTime.map((predTime, index) => (
                            predTime.pred !== 0 ? (
                            <div
                                key={index}
                                time={predTime.time}
                                className="absolute h-full hover:cursor-pointer selection "
                                style={{
                                    width: `${100 / size}%`,
                                    left: `${predTime.progress}%`,
                                    backgroundColor: "yellow",
                                }}
                                onClick={(e) => playVideo(e.target)}
                                    title={index + 1}></div>)
                                : null
                        ))}
                    </div>
                ) : null}
                <div className="h-2/4 w-full relative bg-slate-500">
                    {actions
                        ? Object.keys(actions).map((key, index) => {
                              return actions[key].map((action, index) => {
                                  return (
                                      <div
                                          key={index}
                                          time={action.start_time}
                                          endtime={action.end_time}
                                          className="absolute h-full hover:cursor-pointer selection "
                                          style={{
                                              width: `${
                                                  ((action.end_time - action.start_time) / time) *
                                                  100
                                              }%`,
                                              left: `${(action.start_time / time) * 100}%`,
                                              backgroundColor: "orange",
                                          }}
                                          onClick={(e) => playVideo(e.target)}
                                          title={key}></div>
                                  );
                              });
                          })
                        : null}
                </div>
            </div>
        </>
    );
};

export default Timeline;
