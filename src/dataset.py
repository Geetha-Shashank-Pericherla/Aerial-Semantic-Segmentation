import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
from PIL import Image
from glob import glob

class AerialDataset(Dataset):
    def __init__(self):
        self.imgpath_list = sorted(glob('path_to_original_images/*.jpg'))
        self.labelpath_list = sorted(glob('path_to_label_images/*.png'))

    def __getitem__(self, i):
        imgpath = self.imgpath_list[i]
        img = cv2.imread(imgpath)
        img = cv2.resize(img, (256, 256))
        img = img / 255.0
        img = torch.from_numpy(img.astype(np.float32)).permute(2, 0, 1)

        labelpath = self.labelpath_list[i]
        label = Image.open(labelpath)
        label = np.asarray(label)
        label = cv2.resize(label, (256, 256))
        label = torch.from_numpy(label.astype(np.int64))

        return img, label

    def __len__(self):
        return len(self.imgpath_list)
