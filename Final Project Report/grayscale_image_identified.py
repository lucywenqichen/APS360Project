import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import matplotlib.pyplot as plt
import h5py

from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from tqdm import tqdm
from skimage.color import lab2rgb
from skimage import color, io

from google.colab import drive
drive.mount('/content/drive')

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # Fixes a bug with PyTorch and MKL

# New: We now want to remove all cases where the image is already in grayscale.


grayscale_images = []

for i in tqdm(range(0, 90000)):
  filename = f"{i:07}.png"
  image_path = os.path.join("/content/drive/MyDrive/APS360/Project_Data/lhq_256", filename)
  try:
    image = io.imread(image_path)
    lab = color.rgb2lab(image).astype(np.float16)
    ab = lab[:, :, 1:] / 128.0

  except Exception as e:
    print(f"Error processing image {filename}: {e}")
    break

  if np.max(np.abs(ab)) <= 1e-4:
    grayscale_images.append(i)

print(grayscale_images)
