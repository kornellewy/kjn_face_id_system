import unittest

import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

from kjn_face_id_system.id_card_classification.dataset import IdCardDataset


class IdCaseKFISImageTests(unittest.TestCase):
    def setUp(self) -> None:
        dataset_path = "datasets/id_card_by_cuntry/"
        self.image_size = 100
        img_transform = A.Compose(
            [
                A.Resize(self.image_size, self.image_size),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )
        self.dataset = IdCardDataset(
            dataset_path=dataset_path, img_transform=img_transform
        )

    def test_getitem(self):
        image1, image2, class_idx = self.dataset[0]
        self.assertEqual(image1.shape[0], 3)
        self.assertEqual(image1.shape[1], self.image_size)
        self.assertEqual(image1.shape[2], self.image_size)
        self.assertEqual(image2.shape[0], 3)
        self.assertEqual(image2.shape[1], self.image_size)
        self.assertEqual(image2.shape[2], self.image_size)
        self.assertIn(class_idx, [0, 1])


if __name__ == "__main__":
    unittest.main()
