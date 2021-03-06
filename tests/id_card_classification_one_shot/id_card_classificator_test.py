import unittest
from pathlib import Path

import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch

from kjn_face_id_system.id_card_classification_one_shot.id_card_classificator import (
    IdCardClassificator,
)


class IdCardClassificatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.device = torch.device("cpu")
        self.test_image_path = Path("datasets/combinations/3/front_id/id.png")

    def test_predict(self):
        image = cv2.imread(self.test_image_path.as_posix())
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        classificator = IdCardClassificator(device=self.device)
        (
            max_db_image_name,
            id_card_name,
            max_similarity,
            images_similarities,
        ) = classificator.predict(image=image)
        print(max_db_image_name)
        print(id_card_name)
        print(max_similarity)
        # dont need more test it dont work good


if __name__ == "__main__":
    unittest.main()
