import random
from random import uniform
import torch
import torchvision.transforms.functional as tf
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

        cap = cv2.VideoCapture(clip_path)
        frames = []
        while len(frames) < self.clip_length:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            #Normalised Frames
            frame = frame/255
            frame = torch.tensor(frame, dtype=torch.float32).permute(2, 0, 1)    #Converts OpenCV to PyTorch format

            frames.append(frame)
        cap.release()


        #Safety check for correct frame number so that no errors where made
        if len(frames) < self.clip_length:
            print("WARNING: Too few frames in video clip:" +os.path.basename(clip_path))

        #Apply the same transform to the whole video
        if self.transform:
            frames = self.get_frames_same_transform(frames)

        video_tensor = torch.stack(frames)
        label = torch.tensor(label, dtype = torch.float32)
        return video_tensor, label

    def get_frames_same_transform(self, frames):
        """
        Applies the same set of random transformations to all frames in a video clip.

        :param list[tensors] frames: List of video frames already converted to tensors.
        :return: list[tensors] Transformed frames.
        """

        modified_frames = []

        hflip = random.random() < 0.5    #50% probability for horizontal flip
        greysc = random.random() < 0.3    #30% probability for grayscale
        rotate = random.random() < 0.3    #30% probability for rotation
        crop_resize = random.random() < 0.4    #40% probability for random crop & resize
        brightness_contrast = random.random() < 0.3    #30% probability for brightness/contrast adjustment
        blur = random.random() < 0.2    #20% probability for Gaussian blur

        for frame in frames:
            frame = self.transform(frame)

            #Grayscale
            if greysc:
                frame = tf.rgb_to_grayscale(frame)
                frame = frame.repeat(3, 1, 1)    #Repeat the greyscale frames 3 times, so it is compatible with RGB format

            #Horizontal flip
            if hflip:
                frame = tf.hflip(frame)

            #Rotation (max ±15°)
            if rotate:
                angle = uniform(-15, 15)
                frame = tf.rotate(frame, angle)

            #Random crop & resize
            if crop_resize:
                C, H, W = frame.shape
                new_h = int(H*0.8)
                new_w = int(W*0.8)
                top = random.randint(0, H-new_h)
                left = random.randint(0, W-new_w)
                cropped = tf.crop(frame, top, left, new_h, new_w)
                frame = tf.resize(cropped, [H, W])


            #Brightness & contrast adjustment
            if brightness_contrast:
                frame = tf.adjust_brightness(frame, uniform(0.8, 1.2))
                frame = tf.adjust_contrast(frame, uniform(0.8, 1.2))

            #Gaussian blur
            if blur:
                frame = tf.gaussian_blur(frame, kernel_size=3)

            modified_frames.append(frame)

        return modified_frames
