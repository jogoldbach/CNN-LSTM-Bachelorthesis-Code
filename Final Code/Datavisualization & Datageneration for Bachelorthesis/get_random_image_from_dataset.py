import random
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

from torchvision.utils import make_grid
from video_dataset import Dataset_BA
from video_dataset_adverserial import Dataset_BA as Dataset_BA_adverserial

"""
This script was used to make a 4x3 grid of example images of the dataset.

"""
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((225, 400)),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(0.5),
    transforms.ToTensor()
])

#Load dataset
root_dir = "DIRECTORY TO DATASET"
dataset = Dataset_BA_adverserial(root_dir, transform=transform)

num_samples = 12
random_indices = random.sample(range(len(dataset)), num_samples)

#Extract the first frame from each selected video
frames = []
for idx in random_indices:
    video_tensor, _ = dataset[idx]
    first_frame = video_tensor[0]
    frames.append(first_frame)

grid_img = make_grid(frames, nrow=4, padding=2, normalize=True)

#Convert tensor to NumPy and display
plt.figure(figsize=(10, 8))
plt.imshow(np.transpose(grid_img.numpy(), (1, 2, 0)))
plt.axis("off")
plt.title("Die ersten Bilder von 12 zufälligen Videos nach dem Vorverarbeiten mit Pertubations.")
plt.savefig('visualized_data_examples_pertubations.png', dpi=300, bbox_inches='tight')
plt.show()