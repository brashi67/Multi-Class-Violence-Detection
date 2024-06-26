import os
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
import subprocess
from torch.utils.data import DataLoader
import torch
from model import Model
from dataset import Dataset
import option


app = FastAPI()

# Create the "output/video" directory if it doesn't exist
os.makedirs("input/video", exist_ok=True)

@app.post("/upload_video/")
async def upload_video(file: UploadFile = File(...)):
    try:
        contents = await file.read()  # Read the contents of the file
        video_path = f"input/video/{file.filename}"  
        with open(video_path, "wb") as f:
            f.write(contents)  # Write the contents to the output directory

        # Execute the Python scripts
        execute_script("i3d", video_path) 
        execute_script("vggish", video_path)

        # Make prediction
        scores = make_prediction(video_path)

        return {"filename": file.filename, "scores": scores}
    except Exception:
        return {"message": "There was an error uploading the video"}
    
def execute_script(feature_type, video_path):
    command = [
        "python", 
        "main.py", 
        f"feature_type={feature_type}",
        "device=cpu",
        "on_extraction=save_numpy",
        f"video_paths={video_path}",
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

def make_prediction(video_path):
    args = option.parser.parse_args()
    device = torch.device("cpu")

    test_loader = DataLoader(Dataset(args, test_mode=True),
                              batch_size=5, shuffle=False,
                              num_workers=args.workers, pin_memory=True)
    model = Model(args)
    model = model.to(device)
    model_dict = model.load_state_dict(
        {k.replace('module.', ''): v for k, v in torch.load('ckpt/Binary Inverse Adam/wsanodet_Adam_Binary_Inverse_50.pkl').items()})
    
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
        return probabilities

@app.get("/videos/{video_name}")
async def get_video(video_name: str):
    video_path = f"output/video/{video_name}"
    if os.path.exists(video_path):
        return FileResponse(video_path)
    else:
        return {"message": "Video not found"}

