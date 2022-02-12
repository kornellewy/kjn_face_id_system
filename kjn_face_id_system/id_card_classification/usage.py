import os
from pathlib import Path
from PIL import Image

import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms
import pytorch_lightning as pl

from dataset_multiplefaces import DatasetMultipleFaces
from train_pytorch_lightning import SiameseModule


def convert_model():
    kjn = SiameseModule()
    model_save_path = "face_one_shot_learing/final.pth"
    checkpoint_save_path = str(Path(__file__).parent)
    checkpoint_path = "face_one_shot_learing/lightning_logs/version_5/checkpoints/epoch=98-step=73175.ckpt"
    trainer = pl.Trainer(
        gpus=1,
        benchmark=True,
        max_epochs=10000,
        default_root_dir=checkpoint_save_path,
        check_val_every_n_epoch=1,
        resume_from_checkpoint=checkpoint_path,
        limit_train_batches=0,
        limit_val_batches=0,
        limit_test_batches=0,
    )
    trainer.fit(kjn)
    kjn.conver_model_to_pth(model_save_path)


def usage():
    model_save_path = "face_one_shot_learing/final.pth"
    img1_path = "J:/yt_image_dataset_maker/face_dataset/dr_przemyslaw_kwiecien/image_name_08464_bbox_idx_0_bboxcord_771.56,185.06,1108.78,637.23.jpg"
    img2_path = "J:/yt_image_dataset_maker/face_dataset/scifun/image_name_00021_bbox_idx_0_bboxcord_482.63,82.52,960.35,726.68.jpg"
    img_transforms = transforms.Compose(
        [
            transforms.Resize((100, 100)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    model = torch.load(model_save_path)

    image1 = Image.open(img1_path).convert("RGB")
    image_tensor1 = img_transforms(image1)
    image_tensor1 = image_tensor1.unsqueeze(0)

    image2 = Image.open(img2_path).convert("RGB")
    image_tensor2 = img_transforms(image2)
    image_tensor2 = image_tensor2.unsqueeze(0)

    pred = model(image_tensor2, image_tensor2)
    # same = 1, difrent = 0

    print(pred)


if __name__ == "__main__":
    usage()
