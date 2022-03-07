import os
from pathlib import Path
from PIL import Image
from typing import List
import numpy as np

import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms
import pytorch_lightning as pl
from sklearn.metrics import accuracy_score
from facenet_pytorch import MTCNN

from kjn_face_id_system.id_card_classification.dataset import DatasetMultipleFaces
from train_pytorch_lightning import SiameseModule
from utils import load_files_with_given_extension
from model import Siamese


NO_FACE_DIR_PATHS = ["J:/pufne/Konverotwane/7", "J:/pufne/Konverotwane/20"]
PEOPLE_WITH_ID_CARD_IN_HEAND = [
    "J:/pufne/Konverotwane/17",
    "J:/pufne/Konverotwane/12",
    "J:/pufne/Konverotwane/9",
    "J:/pufne/Konverotwane/16",
    "J:/pufne/Konverotwane/3",
    "J:/pufne/Konverotwane/1",
]


def find_faces(
    dir_path: str,
    face_dir_path: str,
    device: torch.device = torch.device("cpu"),
) -> List[str]:
    mtcnn = MTCNN(keep_all=True, device=device)
    images_paths = load_files_with_given_extension(dir_path)
    face_dir_path = Path(face_dir_path)
    face_dir_path.mkdir(exist_ok=True)
    faces_images_paths = []
    for image_path in images_paths:
        image_name = Path(image_path).stem
        image = Image.open(image_path)
        bboxes, _ = mtcnn.detect(image)
        if isinstance(bboxes, np.ndarray):
            for bbox_idx, bbox in enumerate(bboxes):
                face_bbox = image.crop(bbox)
                bbox_str = ",".join(["{:.2f}".format(x) for x in bbox])
                face_bbox_path = face_dir_path.joinpath(
                    f"image_name_{image_name}_bbox_idx_{bbox_idx}_bboxcord_{bbox_str}.jpg"
                )
                face_bbox.save(face_bbox_path)
                faces_images_paths.append(face_bbox_path)
                # brak bd we want only bigest face on image
    return faces_images_paths


def test2():
    """
    https://github.com/fangpin/siamese-pytorch/blob/master/train.py
    """
    device = torch.device("cpu")
    test_dataset_path = "test"
    img_transform = A.Compose(
        [
            A.Resize(100, 100),
            # A.HorizontalFlip(p=0.5),
            # A.VerticalFlip(p=0.5),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )
    test_dataset = DatasetMultipleFaces(
        dataset_path=test_dataset_path,
        img_transform=img_transform,
    )
    test_dataloader = DataLoader(test_dataset, batch_size=1)
    model = Siamese().to(device)
    model = SiameseModule.load_from_checkpoint(
        "face_one_shot_learing/lightning_logs/version_22/checkpoints/epoch=162-step=159279.ckpt"
    ).to(device)
    pred_list = []
    label_list = []
    for i, data in enumerate(test_dataloader):
        if i > 100:
            break
        img0, img1, label = data
        img0, img1, label = img0.to(device), img1.to(device), label.to(device)
        pred = model(img0, img1)
        pred = pred.data.cpu().numpy()[0][0]
        pred = 1 if pred > 0 else 0
        pred_list.append(pred)
        label = int(label.data.cpu().numpy()[0][0])
        label_list.append(label)
        img0 = img0.cpu().squeeze(0).permute(1, 2, 0).numpy() * 255
        img0 = cv2.cvtColor(img0, cv2.COLOR_RGB2BGR)
        img1 = img1.cpu().squeeze(0).permute(1, 2, 0).numpy() * 255
        img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)
        combo = np.hstack((img0, img1))
        cv2.imwrite(f"{test_dataset_path}/{label}_{i}.jpg", combo)
    acc = accuracy_score(label_list, pred_list)
    print(acc)


if __name__ == "__main__":
    test2()
