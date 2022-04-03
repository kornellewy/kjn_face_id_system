from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

from kjn_face_id_system.id_card_classification.train import Train
from kjn_face_id_system.id_card_classification.utils import (
    load_files_with_given_extension,
)
from kjn_face_id_system.id_card_classification.constans import TYPE_TO_IMAGE_NAME_MAP


class IdCardClassificator:
    WEIGHT_PATH = (
        Path(__file__).parent
        / "lightning_logs/version_0/checkpoints/epoch=199-step=18399.ckpt"
    )
    DATABASE_PATH = "datasets/multiple_types/train/images"

    def __init__(
        self,
        device: Optional[torch.device] = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
    ) -> None:
        self.device = device
        module = Train()
        self.model = module.load_from_checkpoint(self.WEIGHT_PATH.as_posix())
        self.model.to(device)
        self.idx_to_class_name_map = {}
        self.id_card_databese = {}
        self.img_transform = A.Compose(
            [
                A.Resize(224, 224),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )
        self.database = self.populate_database()

    def populate_database(self) -> dict:
        images_paths = load_files_with_given_extension(self.DATABASE_PATH)
        database = {}
        for image_path in images_paths:
            image_name = Path(image_path).stem
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            image = self.img_transform(image=image)["image"]
            image = image.unsqueeze(0)
            with torch.no_grad():
                image_features = self.model.model.forward_one(image)

            database[image_name] = image_features
        return database

    def predict(self, image: np.ndarray) -> Tuple[str, float, dict]:

        image = self.img_transform(image=image)["image"]
        image = image.unsqueeze(0)
        with torch.no_grad():
            image_features = self.model.model.forward_one(image)

        images_similarities = {}
        max_similarity = -1000
        max_db_image_name = ""
        for db_image_name, db_image_features in self.database.items():
            with torch.no_grad():
                similarity = self.model.model.forward_head(
                    db_image_features, image_features
                )
                similarity = similarity.item()
                images_similarities[db_image_name] = similarity
                if similarity > max_similarity:
                    max_similarity = similarity
                    max_db_image_name = db_image_name
        id_card_name = TYPE_TO_IMAGE_NAME_MAP[max_db_image_name]
        return max_db_image_name, id_card_name, max_similarity, images_similarities
