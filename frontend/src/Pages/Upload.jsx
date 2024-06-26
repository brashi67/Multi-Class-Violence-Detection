import { useState, useEffect } from "react";
import { useDropzone } from "react-dropzone";
import { useDispatch } from "react-redux";
import { setResults } from "../state/resultSlice";
import { useNavigate } from "react-router-dom";
import Loader from "../components/Loader";
import axios from "axios";

const Upload = () => {
    const [files, setFiles] = useState([]);
    const [GTFile, setGTFile] = useState(null);
    const [isuploaded, setIsuploaded] = useState(false);
    const [loading, setLoading] = useState(false);
    const dispatch = useDispatch();
    const navigate = useNavigate();

    const { getRootProps, getInputProps } = useDropzone({
        accept: {
            "video/mp4": [],
        },
        onDrop: (acceptedFiles) => {
            setIsuploaded(true);
            setFiles(
                acceptedFiles.map((file) =>
                    Object.assign(file, {
                        preview: URL.createObjectURL(file),
                    })
                )
            );
        },
    });

    async function uploadVideo() {
        try {
            const formData = new FormData();
            let isGT = false;
            formData.append("file", files[0]);
            if (GTFile) {
                isGT = true; 
                formData.append("groundTruth", GTFile);
            }

            const config = {
                method: "post",
                url: "http://127.0.0.1:8000/upload_video/",
                headers: {
                    "Content-Type": "multipart/form-data",
                },
                data: formData,
            };

            setLoading(true);
            const response = await axios(config);
            setLoading(false);
            console.log(response.data);
            dispatch(
                setResults({
                    video: files[0].preview,
                    action: response.data.action,
                    time: response.data.time,
                    gtValues: response.data.groundTruth,
                    IsGT: isGT,
                })
            );
            navigate("/results");
        } catch (error) {
            setLoading(false);
            console.error(error);
        }
    }

    const removeFile = (e) => {
        setIsuploaded(false);
        e.preventDefault();
        e.stopPropagation();
        setFiles([]);
    }    

    const thumbs = files.map((file) => (
        <div className="w-full h-full z-10 flex flex-col justify-center items-center" key={file.name}>
                <video
                    src={file.preview}
                    controls
                    className="rounded-xl rounded-tr-none"
                    // Revoke data uri after image is loaded
                    onLoad={() => {
                        URL.revokeObjectURL(file.preview);
                    }}
                />
        </div>
    ));

    useEffect(() => {
        // Make sure to revoke the data uris to avoid memory leaks, will run on unmount
        return () => files.forEach((file) => URL.revokeObjectURL(file.preview));
    }, []);

    const clearGTInput = (div) => {
        div.parentElement.querySelector("input").value = "";
        setGTFile(null);
    }

    return loading ? (
        <Loader />
    ) : (
        <section className="container flex flex-col items-center gap-5">
            {!isuploaded ? (
                <div {...getRootProps({ className: "dropzone rounded-xl z-10 bg-zinc-700" })}>
                    <input {...getInputProps()} />
                    {files.length ? (
                        thumbs.length > 0 ? (
                            thumbs
                        ) : null
                    ) : (
                        <p>Drag 'n' drop some files here, or click to select files</p>
                    )}
                </div>
            ) : (
                <div className="video">
                    {files.length ? (
                        thumbs.length > 0 ? (
                            thumbs
                        ) : null
                    ) : (
                        <p>Drag 'n' drop some files here, or click to select files</p>
                    )}
                </div>
            )}

                <div className="relative flex items-center transition-all z-20 gap-2 h-20">
                    <div className="gtLabel">
                        Ground Truth File (Optional)
                    </div>
                <div className="relative cursor-pointer bg-transparent flex w-80 justify-between pl-3 items-center h-full rounded-xl border-2 hover:border-blue-400 transition-all border-gray-400">
                    <input
                        type="file"
                        name="GroundTruthFile"
                        id="GroundTruthFile"
                        onChange={(e) => setGTFile(e.target.files[0])}
                        accept=".npy"
                    />
                </div>
                {GTFile ? (
                    <div
                        className="border-2 flex justify-center items-center rounded-xl text-red-600 font-bold text-xl hover:bg-red-600 h-full hover:text-white border-red-600  w-10 cursor-pointer"
                        onClick={(e) => clearGTInput(e.target)}>
                        X
                    </div>
                ) : null}
            </div>

            {files.length ? (
                thumbs.length > 0 ? (
                    <div className="flex justify-center items-center gap-5">
                        <button type="button" onClick={() => uploadVideo()}>
                            {" "}
                            Analyze ⬆️{" "}
                        </button>
                        <button type="button" onClick={(e) => removeFile(e)}>
                            Remove ❌
                        </button>
                    </div>
                ) : null
            ) : null}
        </section>
    );
}

export default Upload