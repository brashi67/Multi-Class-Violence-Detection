from io import BytesIO
import os
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import subprocess
from torch.utils.data import DataLoader
import torch
from MultiModel.model import Model
from MultiModel.dataset import Dataset
from option import Options
import shutil
import numpy as np

app = FastAPI()

origins = [
    "http://localhost:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# Create the "output/video" directory if it doesn't exist
os.makedirs("input/video", exist_ok=True)

@app.post("/upload_video/")
async def upload_video(file: UploadFile = File(...), groundTruth: UploadFile = File(None)):
    try:
        contents = await file.read()  # Read the contents of the file
        video_path = f"input/video/{file.filename}"  
        with open(video_path, "wb") as f:
            f.write(contents)  # Write the contents to the output directory

        GTcontents = []
        if groundTruth:
            GTcontents = await groundTruth.read() # Read the contents of the file
            GTcontents = np.load(BytesIO(GTcontents), allow_pickle=True)

        abspath = os.path.abspath(os.getcwd())

        for directory in ["FlowTest", "RGBTest"]: # "vggish"
            for filename in os.listdir(os.path.join(abspath, "MultiModel/Features", directory)):
                file_path = os.path.join(abspath, "MultiModel/Features", directory, filename)
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)

        # execute_script("i3d", video_path) 
        move_and_copy_files(os.path.join(abspath, "MultiModel/Features/i3d"), os.path.join(abspath, "MultiModel/Features/RGBTest"), os.path.join(abspath, "MultiModel/Features/FlowTest"))
        
        # execute_script("vggish", video_path)
        append_filenames_to_list(os.path.join(abspath, "MultiModel/Features/FlowTest"), os.path.join(abspath, "MultiModel/list/my_Flowtest.list"))
        append_filenames_to_list(os.path.join(abspath, "MultiModel/Features/RGBTest"), os.path.join(abspath, "MultiModel/list/my_RGBtest.list"))
        append_filenames_to_list(os.path.join(abspath, "MultiModel/Features/vggish"), os.path.join(abspath, "MultiModel/list/my_Audiotest.list"))
        
        # Make prediction
        timeStamp, time = make_prediction()

        #remove vggish and i3d files\ ["vggish", "i3d"]
        # for directory in ["i3d"]:
        #     for filename in os.listdir(os.path.join(abspath, "MultiModel/Features", directory)):
        #         file_path = os.path.join(abspath, "MultiModel/Features", directory, filename)
        #         if os.path.isfile(file_path):
        #             os.unlink(file_path)
        #         elif os.path.isdir(file_path):
        #             shutil.rmtree(file_path)

        return {"filename": file.filename, "action": timeStamp, "time": time, "groundTruth": GTcontents.tolist() if len(GTcontents) > 0 else []}
        # return {"message": "Video uploaded successfully", "action": {
        #     "fighting": [],
        #     "shooting": [
        #         { "start_time": 0.0, "end_time": 12.0 },
        #         { "start_time": 17.33, "end_time": 23.33 },
        #     ],
        #     "explosion": [],
        #     "riot": [],
        #     "abuse": [],
        #     "accident": [],
        # }, "time": 60}
    except Exception as e:
        return {"message": "There was an error uploading the videos" + str(e)}
    finally:
        print('Done')
        os.remove(video_path)
    
def execute_script(feature_type, video_path):
    command = [
        "python", 
        "main.py", 
        f"feature_type={feature_type}",
        "device=cuda",
        "on_extraction=save_numpy",
        f"video_paths={video_path}",
        f"output_path=MultiModel/Features/",
        "stack_size=16",  # Make sure these are consistent 
        "step_size=16"   # with your 'main.py' requirements 
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"Script execution for {feature_type} completed successfully.")
        else:
            print(f"Script execution for {feature_type} failed. Error: {result.stderr}")
    except subprocess.CalledProcessError as e:
        print(f"Script execution for {feature_type} failed. Error: {e}")


@app.get("/videos/{video_name}")
async def get_video(video_name: str):
    video_path = f"output/video/{video_name}"
    if os.path.exists(video_path):
        return FileResponse(video_path)
    else:
        return {"message": "Video not found"}


def make_prediction():
    args = Options()
    device = torch.device("cuda")
    abspath = os.path.abspath(os.getcwd())

    test_loader = DataLoader(Dataset(args, test_mode=True),
                              batch_size=5, shuffle=False,
                              num_workers=args.workers, pin_memory=True)
    model = Model(args)
    model = model.to(device)
    model_dict = model.load_state_dict(
        {k.replace('module.', ''): v for k, v in torch.load(os.path.join(abspath,'MultiModel/ckpt/Binary Normal Adam/wsanodetV5.pkl') ).items()})
    
    model.eval()
    with torch.no_grad():
        pred = torch.zeros(0).to(device)
        for i, input in enumerate(test_loader):
            input = input.to(device)
            logits, _ = model(inputs=input, seq_len=None)
            probs = torch.softmax(logits, 2)
            probs = torch.mean(probs, dim=0)
            pred = torch.argmax(probs, 1).float()

        probabilities = pred.cpu().detach().numpy().tolist()
        probabilities = [round(num, 3) for num in probabilities]
        
        time_stramp = {
            "fighting":[],
            "shooting":[],
            "explosion":[],
            "riot":[],
            "abuse":[],
            "accident":[]
        }

        occurence = 0

        activity = set(probabilities)

        for i in activity:
            for j in range(len(probabilities)-1):
                if(i == probabilities[j] and occurence == 0):
                    start_frame = j
                    start_timestamp = (start_frame * 16)/ 24
                if(i == probabilities[j]):
                    occurence = occurence + 1
                if( (i == probabilities[j] and i != probabilities[j+1]) or ( i == probabilities[j] and j == len(probabilities)-2 )):
                    occurence = 0
                    end_frame = j
                    end_timestamp = (end_frame * 16)/ 24
                    if(end_timestamp - start_timestamp < 3):
                        continue
                    if(i==1):
                        time_stramp["fighting"].append({"start_time":round(start_timestamp, 2),"end_time":round(end_timestamp, 2)})
                    elif(i==2):
                        time_stramp["shooting"].append({"start_time":round(start_timestamp, 2),"end_time":round(end_timestamp, 2)})
                    elif(i==3):
                        time_stramp["explosion"].append({"start_time":round(start_timestamp, 2),"end_time":round(end_timestamp, 2)})
                    elif(i==4):
                        time_stramp["riot"].append({"start_time":round(start_timestamp, 2),"end_time":round(end_timestamp, 2)})
                    elif(i==5):
                        time_stramp["abuse"].append({"start_time":round(start_timestamp, 2),"end_time":round(end_timestamp, 2)})
                    elif(i==6):
                        time_stramp["accident"].append({"start_time":round(start_timestamp, 2),"end_time":round(end_timestamp, 2)})

        print(probabilities)
        print(time_stramp)
        time = (len(probabilities)*16)/24
        return time_stramp, time 

def append_filenames_to_list(directory_path, list_file_path):
    try:
        # Clear the contents of the list file
        with open(list_file_path, "w") as f:
            f.write("")

        # Get the list of filenames in the directory
        filenames = os.listdir(directory_path)
        
        # Append each filename to the list file
        with open(list_file_path, "a") as f:
            for filename in filenames:
                f.write(os.path.join(directory_path, filename) + "\n")
        
        return {"message": "File names appended successfully"}
    except Exception as e:
        return {"message": f"Error: {str(e)}"}
    
def move_and_copy_files(i3d_directory, rgb_directory, flow_directory):
    try:
        # Get list of files in the "i3d" directory
        files = os.listdir(i3d_directory)
        print(files)

        # Look for RGB and Flow files and move them to their respective directories and make 5 copies
        for filename in files:
            if "rgb" in filename.lower():
                # Move the original RGB file
                shutil.move(os.path.join(i3d_directory, filename), os.path.join(rgb_directory, filename))
                # Create 5 copies with index 1, 2, 3, 4, and 5
                for i in range(0, 5):
                    copy_filename = filename.rsplit('.', 1)[0] + f"_{i}" + "." + filename.rsplit('.', 1)[-1]
                    shutil.copy(os.path.join(rgb_directory, filename), os.path.join(rgb_directory, copy_filename))
                os.remove(os.path.join(rgb_directory, filename))
            elif "flow" in filename.lower():
                # Move the original Flow file
                shutil.move(os.path.join(i3d_directory, filename), os.path.join(flow_directory, filename))
                # Create 5 copies with index 1, 2, 3, 4, and 5
                for i in range(0, 5):
                    copy_filename = filename.rsplit('.', 1)[0] + f"_{i}" + "." + filename.rsplit('.', 1)[-1]
                    shutil.copy(os.path.join(flow_directory, filename), os.path.join(flow_directory, copy_filename))
                os.remove(os.path.join(flow_directory, filename))
        
        return {"message": "RGB and Flow files moved and copied successfully"}
    except Exception as e:
        return {"message": f"Error: {str(e)}"}