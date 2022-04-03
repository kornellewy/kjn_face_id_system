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

from kjn_face_id_system.id_card_classification_one_shot.dataset import IdCardDataset
from kjn_face_id_system.id_card_classification_one_shot.model import Siamese
from kjn_face_id_system.optims.Adam import AdamW_GCC2


class Train(pl.LightningModule):
    def __init__(
        self,
        hparams={
            "batch_size": 64,
            "lr": 0.00006,
            "dataset_path": "datasets/fake_id_cards_for_oneshot",
        },
    ):
        super().__init__()
        self._hparams = hparams
        self.model = Siamese()
        self.criterion = torch.nn.BCEWithLogitsLoss(size_average=True)
        self.img_transform = A.Compose(
            [
                A.Resize(224, 224),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )
        self.train_set, self.valid_set, self.test_set = self.split_datasets()

    def split_datasets(self):
        dataset = IdCardDataset(
            self._hparams["dataset_path"], img_transform=self.img_transform
        )
        train_size = int(0.98 * len(dataset))
        valid_size = int(0.01 * len(dataset))
        test_size = int(0.01 * len(dataset))
        rest = len(dataset) - train_size - valid_size - test_size
        train_size = train_size + rest
        train_set, valid_set, test_set = torch.utils.data.random_split(
            dataset, [train_size, valid_size, test_size]
        )
        return train_set, valid_set, test_set

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.hparams["batch_size"])

    def val_dataloader(self):
        return DataLoader(self.valid_set, batch_size=self.hparams["batch_size"])

    def test_dataloader(self):
        return DataLoader(self.train_set, batch_size=1)

    def configure_optimizers(self):
        optimizer = AdamW_GCC2(self.model.parameters(), lr=self.hparams["lr"])
        return optimizer

    def forward(self, img0, img1):
        return self.model(img0, img1)

    def forward_single(self, img):
        return self.model.forward_one(img)

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
        return loss, pred.detach()

    def validation_epoch_end(self, batches):
        pred_list = []
        label_list = []
        for batch in batches:
            pred, label = batch
            if pred.shape == torch.Size([]):
                return
            pred = pred.data.cpu().numpy()[0][0]
            pred = 1 if pred > 0 else 0
            label = int(label.data.cpu().numpy()[0][0])
            pred_list.append(pred)
            label_list.append(label)
        acc = accuracy_score(label_list, pred_list)
        print(acc)

    def test_step(self, batch, batch_nb):
        img0, img1, label = batch
        pred = self(img0, img1)
        return pred, label

    def test_epoch_end(self, batches):
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


if __name__ == "__main__":
    module = Train()
    checkpoint_save_path = str(Path(__file__).parent)
    # # checkpoint_path = "face_one_shot_learing/lightning_logs/version_24/checkpoints/epoch=20-step=111321.ckpt"
    # trainer = pl.Trainer(
    #     gpus=1,
    #     precision=16,
    #     benchmark=True,
    #     max_epochs=200,
    #     default_root_dir=checkpoint_save_path,
    #     check_val_every_n_epoch=1,
    #     # resume_from_checkpoint=checkpoint_path,
    # )
    # trainer.fit(module)
    # trainer.test(module)

    checkpoint_path = "kjn_face_id_system/id_card_classification/lightning_logs/version_0/checkpoints/epoch=199-step=18399.ckpt"
    module = module.load_from_checkpoint(checkpoint_path)

    trainer = pl.Trainer(
        gpus=1,
        precision=16,
        benchmark=True,
        max_epochs=200,
        default_root_dir=checkpoint_save_path,
        check_val_every_n_epoch=1,
        resume_from_checkpoint=checkpoint_path,
    )
    trainer.test(module)
