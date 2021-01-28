from __future__ import print_function
import torch
from torch.utils.data import Dataset
import glob
import os
import numpy as np
import csv
from collections import Counter
from torchvision import transforms
from PIL import Image
import pandas as pd
from collections import Counter


train_transform = transforms.Compose(
    [transforms.RandomHorizontalFlip(p=0.5),
     transforms.RandomResizedCrop(224),
     transforms.ToTensor(),
     # transforms.Normalize([0.485], [0.229])
     ])
test_transform = transforms.Compose(
    [transforms.Resize([256, 256]),
     transforms.CenterCrop([224, 224]),
     transforms.ToTensor(),
     # transforms.Normalize([0.485], [0.229])
     ])


class Retina_Dataset_Simple(Dataset):
    def __init__(self, samples, labels, args, mode='train'):
        super(Retina_Dataset_Simple, self).__init__()
        self.args = args
        self.mode = mode
        self.images = {}
        self.counter = Counter()
        for file in samples:
            filename = os.path.basename(os.path.splitext(file)[0])
            self.images[filename] = Image.fromarray(np.load(file))

        self.set = list(self.images.keys())

        # Loading labels
        self.labels = {}
        for row in labels:
            if row['image'] in self.set:
                label = int(int(row['level']) > 0)
                self.counter[label] += 1
                self.labels[row['image']] = label

        print("Created", mode, "Dataloader with ", len(self.set), 'samples', str(self.counter))

    def __getitem__(self, idx):
        key = self.set[idx]

        if self.mode == 'train':
            transform_func = train_transform
        else:
            transform_func = test_transform

        return {'image': transform_func(self.images[key]),
                'label': np.array([self.labels[key]]),
                'img_name': key}

    def __len__(self):
        return len(self.set)

    def set_mode(self, mode):
        self.mode = mode
        return self
