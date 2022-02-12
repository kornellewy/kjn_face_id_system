"""
source: 
https://github.com/cskarthik7/One-Shot-Learning-PyTorch

"""
from pathlib import Path

import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from sklearn.metrics import accuracy_score

from dataset_multiplefaces import DatasetMultipleFaces
from model import Siamese


class SiameseModule(pl.LightningModule):
    def __init__(
        self,
        hparams={
            "batch_size": 256,
            "lr": 0.00006,
            "train_dataset_path": "J:/yt_image_dataset_maker/face_dataset_train",
            "valid_dataset_path": "J:/yt_image_dataset_maker/face_dataset_valid",
            "test_dataset_path": "J:/yt_image_dataset_maker/face_dataset_test",
        },
    ):
        super().__init__()
        self._hparams = hparams
        self.model = Siamese()
        self.criterion = torch.nn.BCEWithLogitsLoss(size_average=True)
        self.img_transform = A.Compose(
            [
                A.Resize(100, 100),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams["lr"])
        return optimizer

    def forward(self, img0, img1):
        return self.model(img0, img1)

    def training_step(self, batch, batch_nb):
        img0, img1, label = batch
        pred = self(img0, img1)
        loss = self.criterion(pred, label)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_nb):
        img0, img1, label = batch
        pred = self(img0, img1)
        loss = self.criterion(pred, label)
        self.log("valid_loss", loss)
        return loss

    def test_step(self, batch, batch_nb):
        img0, img1, label = batch
        pred = self(img0, img1)
        return pred, label

    def test_epoch_end(self, batches):
        all_test_step_outs = batches
        pred_list = []
        label_list = []
        for batch in batches:
            pred, label = batch
            pred = pred.data.cpu().numpy()[0][0]
            pred = 1 if pred > 0 else 0
            label = int(label.data.cpu().numpy()[0][0])
            pred_list.append(pred)
            label_list.append(label)
        acc = accuracy_score(label_list, pred_list)
        print(acc)

    def train_dataloader(self):
        dataset = DatasetMultipleFaces(
            dataset_path=self.hparams["train_dataset_path"],
            img_transform=self.img_transform,
        )
        return DataLoader(dataset, batch_size=self.hparams["batch_size"])

    def val_dataloader(self):
        dataset = DatasetMultipleFaces(
            dataset_path=self.hparams["valid_dataset_path"],
            img_transform=self.img_transform,
        )
        return DataLoader(dataset, batch_size=self.hparams["batch_size"])

    def test_dataloader(self):
        dataset = DatasetMultipleFaces(
            dataset_path=self.hparams["test_dataset_path"],
            img_transform=self.img_transform,
        )
        return DataLoader(dataset, batch_size=1)

    def conver_model_to_pth(self, model_path):
        torch.save(self.model, model_path)


if __name__ == "__main__":
    kjn = SiameseModule()
    checkpoint_save_path = str(Path(__file__).parent)
    checkpoint_path = "face_one_shot_learing/lightning_logs/version_22/checkpoints/epoch=162-step=159279.ckpt"
    trainer = pl.Trainer(
        gpus=1,
        precision=16,
        benchmark=True,
        max_epochs=200,
        default_root_dir=checkpoint_save_path,
        check_val_every_n_epoch=1,
        resume_from_checkpoint=checkpoint_path,
    )
    trainer.fit(kjn)
    trainer.test(kjn)
    model_save_path = "face_one_shot_learing/final.pth"
    kjn.conver_model_to_pth(model_save_path)
