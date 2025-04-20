import random

import numpy as np
import torch
import cv2
import os
from torch.utils.data import Dataset


class Dataset_BA(Dataset):
    #https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
    def __init__(self, root_dir, clip_length = 30, transform=None):
        self.root_dir = root_dir
        self.clip_length = clip_length
        self.transform = transform

        self.samples = []
        for label, classification_name in enumerate(["Idle", "Active"]):
            classification_dir = os.path.join(self.root_dir, classification_name)
            for video_clip in os.listdir(classification_dir):
                if video_clip.endswith(".mp4"):
                    self.samples.append((os.path.join(classification_dir, video_clip), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        clip_path, label = self.samples[idx]

        cap = cv2.VideoCapture(clip_path)
        frames = []
        while len(frames) < self.clip_length:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            noise_type = random.choice(["gauss", "blurr", "brightness/contrast"])
            frame = self.make_noisy_and_normalize(noise_type, frame)

            if self.transform:
                frame = self.transform(frame)
            frames.append(frame)
        cap.release()

        #Safety check for correct frame number so that no errors where made
        if len(frames) < self.clip_length:
            print("WARNING: Too few frames in video clip:" + os.path.basename(clip_path))

        #Put frames into tensor
        video_tensor = torch.stack(frames)
        label = torch.tensor(label, dtype=torch.float32)
        return video_tensor, label


    def make_noisy_and_normalize(self, noise_typ, image):
        """
        Applies a selected noise type to an image.

        :param string noise_typ: The type of noise to apply.
        :param image: The input RGB image.
        :return: The noisy, normalized image in float format.
        """

        #Normalised Frames
        image = image / 255

        if noise_typ == "gauss":    #Gaussian noise
            mean = 0
            var = 0.02
            sigma = var ** 0.5
            gauss = np.random.normal(mean, sigma, image.shape).astype(np.float32)
            image = np.clip(image + gauss, 0, 1)
            return image

        elif noise_typ == "blurr":
            kernel_size = 15
            kernel = np.ones((kernel_size,kernel_size), np.float32)/ (kernel_size **2)
            image = cv2.filter2D(image, -1, kernel)    #Mean blur
            return image

        elif noise_typ == "brightness/contrast":
            alpha = np.random.uniform(0.7,1.3)    #Contrast
            beta = np.random.uniform(-40,40)/255.0    #Brightness
            image = np.clip(alpha * image + beta, 0, 1)
            return image

