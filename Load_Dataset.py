# -*- coding: utf-8 -*-

import os
import cv2
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import functional as F
from transformers import BertTokenizer, BertModel

# -----------------------------
# Device
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================
# TEXT ENCODER (Modern BERT)
# =============================
class TextEncoder:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained(
            "bert-base-uncased",
            local_files_only=True
        )

        self.model = BertModel.from_pretrained(
            "bert-base-uncased",
            local_files_only=True
        )
        self.model.to(device)
        self.model.eval()

    def encode(self, text_list):
        inputs = self.tokenizer(
            text_list,
            padding='max_length',
            truncation=True,
            max_length=10,
            return_tensors="pt"
        )

        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        # Get full token embeddings
        embeddings = outputs.last_hidden_state  # (1, 10, 768)

        return embeddings.squeeze(0).cpu()  # (10, 768)


# =============================
# AUGMENTATIONS
# =============================
def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)

    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()

    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = F.rotate(F.to_pil_image(image), angle)
    label = F.rotate(F.to_pil_image(label), angle)
    return np.array(image), np.array(label)


# =============================
# TRAIN GENERATOR
# =============================
class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size
        self.to_tensor = T.ToTensor()

    def __call__(self, sample):
        image, label, text = sample['image'], sample['label'], sample['text']

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)

        image = cv2.resize(image, (self.output_size[0], self.output_size[1]))
        label = cv2.resize(label, (self.output_size[0], self.output_size[1]))

        image = self.to_tensor(image)
        label = torch.from_numpy(label).long()

        return {
            'image': image,
            'label': label,
            'text': text
        }


# =============================
# VALIDATION GENERATOR
# =============================
class ValGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size
        self.to_tensor = T.ToTensor()

    def __call__(self, sample):
        image, label, text = sample['image'], sample['label'], sample['text']

        image = cv2.resize(image, (self.output_size[0], self.output_size[1]))
        label = cv2.resize(label, (self.output_size[0], self.output_size[1]))

        image = self.to_tensor(image)
        label = torch.from_numpy(label).long()

        return {
            'image': image,
            'label': label,
            'text': text
        }


# =============================
# MAIN IMAGE + TEXT DATASET
# =============================
class ImageToImage2D(Dataset):

    def __init__(
        self,
        dataset_path: str,
        task_name: str,
        row_text: dict,
        joint_transform=None,
        image_size: int = 224
    ):
        self.dataset_path = dataset_path
        self.task_name = task_name
        self.image_size = image_size

        self.input_path = os.path.join(dataset_path, 'img')
        self.output_path = os.path.join(dataset_path, 'labelcol')

        self.images_list = os.listdir(self.input_path)
        self.rowtext = row_text

        self.text_encoder = TextEncoder()
        self.joint_transform = joint_transform

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):

        image_filename = self.images_list[idx]
        mask_filename = image_filename[:-3] + "png"

        image = cv2.imread(os.path.join(self.input_path, image_filename))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(os.path.join(self.output_path, mask_filename), 0)
        mask[mask <= 0] = 0
        mask[mask > 0] = 1

        text_raw = self.rowtext[mask_filename]
        text_list = text_raw.split("\n")
        text_embedding = self.text_encoder.encode(text_list)

        sample = {
            'image': image,
            'label': mask,
            'text': text_embedding
        }

        if self.joint_transform:
            sample = self.joint_transform(sample)

        return sample, image_filename


# =============================
# LABEL + TEXT ONLY DATASET
# =============================
class LV2D(Dataset):

    def __init__(
        self,
        dataset_path: str,
        task_name: str,
        row_text: dict,
        joint_transform=None,
        image_size: int = 224
    ):
        self.dataset_path = dataset_path
        self.task_name = task_name
        self.image_size = image_size

        self.output_path = dataset_path
        self.mask_list = os.listdir(self.output_path)
        self.rowtext = row_text

        self.text_encoder = TextEncoder()
        self.joint_transform = joint_transform

    def __len__(self):
        return len(self.mask_list)

    def __getitem__(self, idx):

        mask_filename = self.mask_list[idx]

        mask = cv2.imread(os.path.join(self.output_path, mask_filename), 0)
        mask = cv2.resize(mask, (self.image_size, self.image_size))
        mask[mask <= 0] = 0
        mask[mask > 0] = 1

        text_raw = self.rowtext[mask_filename]
        text_list = text_raw.split("\n")
        text_embedding = self.text_encoder.encode(text_list)

        sample = {
            'label': mask,
            'text': text_embedding
        }

        if self.joint_transform:
            sample = self.joint_transform(sample)

        return sample, mask_filename