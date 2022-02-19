import unittest
from pathlib import Path

import numpy as np

from kjn_face_id_system.images.base_image import BaseKFISImage


class BaseKFISImageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.test_image = Path("test_case_example/selfie/selfie.png")
        self.test_image_height = 576
        self.test_image_width = 1024

    def test_creat_image(self):
        base_image = BaseKFISImage(image_path=self.test_image)
        self.assertIsInstance(base_image.image, np.ndarray)
        self.assertEqual(base_image.height, self.test_image_height)
        self.assertEqual(base_image.width, self.test_image_width)

    def tearDown(self) -> None:
        return super().tearDown()


if __name__ == "__main__":
    unittest.main()
