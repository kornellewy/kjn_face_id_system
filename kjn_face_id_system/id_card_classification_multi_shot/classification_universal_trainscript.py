"""
source:
https://stackoverflow.com/questions/65498782/how-to-dump-confusion-matrix-using-tensorboard-logger-in-pytorch-lightning
https://pytorch-lightning.readthedocs.io/en/latest/starter/introduction_guide.html
"""

import torch
from PIL import Image, ImageFile
from torch.utils.data.sampler import Sampler
from sklearn.metrics import confusion_matrix
from torchvision import datasets, models, transforms
import torchvision
import os
import pytorch_lightning as pl
from pytorch_lightning import Trainer
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path
import torch.nn as nn
import cv2
import numpy as np
import json
from pytorch_lightning.callbacks import ModelCheckpoint

from kjn_face_id_system.optims.Adam import AdamW_GCC2

ImageFile.LOAD_TRUNCATED_IMAGES = True
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
Image.MAX_IMAGE_PIXELS = None


class AlbumentationsTransform:
    def __init__(self, agumentation={}):
        self.img_transforms = A.Compose(
            [
                A.Resize(224, 224),
                A.RGBShift(),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.2),
                A.ChannelShuffle(0.5),
                A.ColorJitter(p=0.3),
                A.Cutout(
                    num_holes=3,
                    max_h_size=24,
                    max_w_size=24,
                    fill_value=0,
                    always_apply=False,
                    p=0.5,
                ),
                A.ShiftScaleRotate(
                    scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0
                ),
                A.PadIfNeeded(
                    min_height=224, min_width=224, always_apply=True, border_mode=0
                ),
                A.IAAAdditiveGaussianNoise(p=0.2),
                A.IAAPerspective(p=0.3),
                A.ShiftScaleRotate(border_mode=cv2.BORDER_CONSTANT, p=0.3),
                A.RandomBrightnessContrast(p=0.3),
                A.OneOf(
                    [
                        A.CLAHE(p=1),
                        A.RandomBrightness(p=1),
                        A.RandomGamma(p=1),
                    ],
                    p=0.3,
                ),
                A.OneOf(
                    [
                        A.IAASharpen(p=1),
                        A.Blur(blur_limit=3, p=1),
                        A.MotionBlur(blur_limit=3, p=1),
                    ],
                    p=0.3,
                ),
                A.OneOf(
                    [
                        A.RandomContrast(p=1),
                        A.HueSaturationValue(p=1),
                    ],
                    p=0.3,
                ),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )

    def __call__(self, img):
        img = np.array(img)
        return self.img_transforms(image=img).copy()


class ImbalancedDatasetSampler(Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices (list, optional): a list of indices
        num_samples (int, optional): number of samples to draw
        callback_get_label func: a callback-like function which takes two arguments - dataset and index
    """

    def __init__(
        self, dataset, indices=None, num_samples=None, callback_get_label=None
    ):
        self.indices = list(range(len(dataset))) if indices is None else indices
        self.callback_get_label = callback_get_label
        self.num_samples = len(self.indices) if num_samples is None else num_samples
        label_to_count = {}
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            if label in label_to_count:
                label_to_count[label] += 1
            else:
                label_to_count[label] = 1
        weights = [
            1.0 / label_to_count[self._get_label(dataset, idx)] for idx in self.indices
        ]
        self.weights = torch.DoubleTensor(weights)

    def _get_label(self, dataset, idx):
        return dataset[idx][1]

    def __iter__(self):
        return (
            self.indices[i]
            for i in torch.multinomial(self.weights, self.num_samples, replacement=True)
        )

    def __len__(self):
        return self.num_samples


class UniversaClassificationTrainer(pl.LightningModule):
    def __init__(
        self,
        hparams={
            "epochs_num": 100,
            "batch_size": 64,
            "lr": 0.0002,
            "train_valid_test_split": [0.9, 0.05, 0.05],
        },
        model_type="resnet50",
        dataset_path="dataset",
        folders_structure={
            "models_folder": str(Path(__file__).parent / "models"),
            "graph_folder": str(Path(__file__).parent / "graph_folder"),
            "confusion_matrix_folder": str(Path(__file__).parent / "confusion_matrix"),
            "test_img_folder": str(Path(__file__).parent / "test_img_folder"),
            "metadata_json_folder": str(Path(__file__).parent / "metadata_json"),
        },
    ):
        super().__init__()
        self.metadata_dict = {}
        self._hparams = hparams
        self.model_type = model_type
        self.dataset_path = dataset_path
        self.model = self._load_specific_model()
        self.img_transforms = transforms.Compose([AlbumentationsTransform()])
        self.criterion = nn.CrossEntropyLoss()
        self.folders_structure = folders_structure
        self._split_dataset_to_dataloaders_and_return_classes()

    def _load_specific_model(self):
        number_of_classes = self._get_number_of_classes()
        if self.model_type == "resnet50":
            model = models.resnet50(pretrained=True).to(self.device)
            # for param in model.parameters():
            #     param.requires_grad = False
            model.fc = nn.Sequential(
                nn.Linear(2048, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, number_of_classes),
            ).to(self.device)
            model = model.to(self.device)
        elif self.model_type == "resnet18":
            model = models.resnet18(pretrained=True).to(self.device)
            for param in model.parameters():
                param.requires_grad = False
            model.fc = nn.Sequential(
                nn.Linear(512, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, number_of_classes),
            ).to(self.device)
            model = model.to(self.device)
        elif self.model_type == "resnet34":
            model = models.resnet34(pretrained=True).to(self.device)
            for param in model.parameters():
                param.requires_grad = False
            model.fc = nn.Sequential(
                nn.Linear(2048, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, number_of_classes),
            ).to(self.device)
            model = model.to(self.device)
        elif self.model_type == "resnet101":
            model = models.resnet101(pretrained=True).to(self.device)
            for param in model.parameters():
                param.requires_grad = False
            model.fc = nn.Sequential(
                nn.Linear(2048, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, number_of_classes),
            ).to(self.device)
            model = model.to(self.device)
        elif self.model_type == "resnet152":
            model = models.resnet152(pretrained=True).to(self.device)
            for param in model.parameters():
                param.requires_grad = False
            model.fc = nn.Sequential(
                nn.Linear(2048, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, number_of_classes),
            ).to(self.device)
            model = model.to(self.device)
        elif self.model_type == "vgg16":
            model = models.vgg16(pretrained=True)
            for param in model.parameters():
                param.requires_grad = False
            model.classifier = nn.Sequential(
                nn.Linear(25088, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, number_of_classes),
            ).to(self.device)
            model = model.to(self.device)
        return model

    def _get_number_of_classes(self):
        return len([f.path for f in os.scandir(self.dataset_path) if f.is_dir()])

    def get_training_augmentation(self):
        return lambda img: self.img_transforms(image=np.array(img))

    def _split_dataset_to_dataloaders_and_return_classes(self):
        dataset = datasets.ImageFolder(dataset_path, self.img_transforms)
        classes = dataset.classes
        self.classes = classes
        class_to_idx = dataset.class_to_idx
        self.class_to_idx = class_to_idx
        train_size = int(self.hparams["train_valid_test_split"][0] * len(dataset))
        valid_size = int(self.hparams["train_valid_test_split"][1] * len(dataset))
        test_size = int(self.hparams["train_valid_test_split"][2] * len(dataset))
        rest = len(dataset) - train_size - valid_size - test_size
        train_size = train_size + rest
        if train_size + valid_size + test_size == len(dataset):
            train_set, valid_set, test_set = torch.utils.data.random_split(
                dataset, [train_size, valid_size, test_size]
            )
        else:
            try:
                train_set, valid_set, test_set = torch.utils.data.random_split(
                    dataset, [train_size + 1, valid_size, test_size]
                )
            except:
                train_set, valid_set, test_set = torch.utils.data.random_split(
                    dataset, [train_size, valid_size + 1, test_size]
                )
        self.train_set = train_set
        self.valid_set = valid_set
        self.test_set = test_set
        self.metadata_dict.update({"classes": classes})
        self.metadata_dict.update({"class_to_idx": class_to_idx})

    def forward(self, x):
        pred = self.model(x)
        return pred

    def train_dataloader(self):
        train_dataloader = torch.utils.data.DataLoader(
            self.train_set,
            sampler=ImbalancedDatasetSampler(self.train_set),
            batch_size=self.hparams["batch_size"],
            num_workers=10,
        )
        return train_dataloader

    def val_dataloader(self):
        val_dataloader = torch.utils.data.DataLoader(
            self.valid_set,
            sampler=ImbalancedDatasetSampler(self.valid_set),
            batch_size=self.hparams["batch_size"],
            num_workers=1,
        )
        return val_dataloader

    def test_dataloader(self):
        test_dataloader = torch.utils.data.DataLoader(
            self.test_set,
            sampler=ImbalancedDatasetSampler(self.test_set),
            batch_size=self.hparams["batch_size"],
            num_workers=1,
        )
        return test_dataloader

    def training_step(self, batch, batch_nb):
        x, y = batch
        if isinstance(x, dict):
            x = x["image"]
        pred = self(x)
        loss = self.criterion(pred, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_nb):
        x, y = batch
        if isinstance(x, dict):
            x = x["image"]
        pred = self(x)
        loss = self.criterion(pred, y)
        self.log("val_loss", loss)
        # Calculate accuracy
        top_p, top_class = pred.topk(1, dim=1)
        equals = top_class == y.view(*top_class.shape)
        accuracy = torch.mean(equals.type(torch.FloatTensor)).item()
        self.log("valid_accuracy", accuracy)
        return loss

    def validation_epoch_end(self, outputs):
        os.makedirs(self.folders_structure["metadata_json_folder"], exist_ok=True)
        self.metadata_dict.update({"hparams": self.hparams})
        self.metadata_dict.update({"model_type": self.model_type})
        json_file_path = os.path.join(
            self.folders_structure["metadata_json_folder"], "metadata.json"
        )
        with open(json_file_path, "w") as fp:
            json.dump(self.metadata_dict, fp)
        return

    def test_step(self, batch, batch_nb):
        x, y = batch
        if isinstance(x, dict):
            x = x["image"]
        pred = self(x)
        loss = self.criterion(pred, y)
        model_export_save_dir_path = os.path.join(
            self.folders_structure["models_folder"]
        )
        os.makedirs(model_export_save_dir_path, exist_ok=True)
        model_name = self.model_type + ".pth"
        model_save_path = os.path.join(model_export_save_dir_path, model_name)
        torch.save(self.model, model_save_path)
        self.log("test_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = AdamW_GCC2(self.model.parameters(), lr=self.hparams["lr"])
        return optimizer


if __name__ == "__main__":
    torch.cuda.empty_cache()
    dataset_path = "datasets/fake_id_cards_for_oneshot"
    model_type = "resnet50"
    model = UniversaClassificationTrainer(
        dataset_path=dataset_path, model_type=model_type
    )
    checkpoint_save_path = str(Path(__file__).parent / "checkpoints_staff")
    n_best_model_save_path = str(Path(__file__).parent / "n_best_model")
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=n_best_model_save_path,
        filename="{epoch:02d}-{val_loss:.2f}",
        save_top_k=1,
        mode="min",
    )
    trainer = Trainer(
        gpus=1,
        precision=16,
        auto_lr_find=True,
        benchmark=True,
        max_epochs=1000,
        default_root_dir=checkpoint_save_path,
        check_val_every_n_epoch=1,
        callbacks=[checkpoint_callback],
    )
    # trainer.tune(model)
    # lr_finder = trainer.tuner.lr_find(model, early_stop_threshold=100)
    # new_lr = lr_finder.suggestion()
    # model.hparams['lr'] = new_lr
    trainer.fit(model)
    trainer.test(model)
