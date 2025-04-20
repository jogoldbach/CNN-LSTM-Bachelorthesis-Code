import torch
import cv2
import os
from torch.utils.data import Dataset


class Dataset_BA(Dataset):
    #https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
    def __init__(self, root_dir, clip_length = 30,transform=None, device='cuda'):
        self.root_dir = root_dir
        self.clip_length = clip_length
        self.transform = transform
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        self.samples = []
        for label, classification_name in enumerate(["Idle","Active"]):
            classification_dir = os.path.join(self.root_dir, classification_name)
            for video_clip in os.listdir(classification_dir):
                if video_clip.endswith(".mp4"):
                    self.samples.append((os.path.join(classification_dir, video_clip), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        clip_path, label = self.samples[idx]

        #Read video and get each frame
        cap = cv2.VideoCapture(clip_path)
        frames = []
        while len(frames) < self.clip_length:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            #Normalised Frames
            frame = frame/255
            frame = torch.tensor(frame, dtype=torch.float32).permute(2, 0, 1)
            if self.transform:
                frame = self.transform(frame)
            frames.append(frame)
        cap.release()


        #Safety check for correct frame number so that no errors where made
        if len(frames) < self.clip_length:
            print("WARNING: Too few frames in video clip:" +os.path.basename(clip_path))

        #Put frames into tensor
        video_tensor = torch.stack(frames)
        label = torch.tensor(label, dtype = torch.float32)
        return video_tensor, label
