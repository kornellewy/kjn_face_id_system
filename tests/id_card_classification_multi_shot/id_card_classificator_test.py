import unittest
from pathlib import Path

import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch

from kjn_face_id_system.id_card_classification_multi_shot.id_card_classificator import (
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
        prediction = classificator.predict(image=image)
        self.assertIsInstance(prediction, dict)

if __name__ == "__main__":
    unittest.main()
