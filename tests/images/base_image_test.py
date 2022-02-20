import unittest
from pathlib import Path

import numpy as np

from kjn_face_id_system.images.base_image import BaseKFISImage
from tests.utils import TEST_TEMP_DIR_PATH


class BaseKFISImageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.test
        self.test_image_path = Path("test_case_example/selfie/id.png")
        self.test_image_height = 576
        self.test_image_width = 1024
        self.test_new_big_image_height = 1000
        self.test_new_big_image_width = 2000
        self.test_new_small_image_height = 1000
        self.test_new_small_image_width = 2000

    def test_creat_image(self):
        base_image = BaseKFISImage(image_path=self.test_image_path)
        self.assertIsInstance(base_image.image, np.ndarray)
        self.assertEqual(base_image.height, self.test_image_height)
        self.assertEqual(base_image.width, self.test_image_width)

    def test_resize_image(self):
        base_image = BaseKFISImage(image_path=self.test_image_path)
        base_image.resize_image(
            new_width=self.test_new_big_image_width,
            new_height=self.test_new_big_image_height,
        )
        self.assertEqual(base_image.height, self.test_new_big_image_height)
        self.assertEqual(base_image.width, self.test_new_big_image_width)
        base_image.resize_image(
            new_width=self.test_new_small_image_width,
            new_height=self.test_new_small_image_height,
        )
        self.assertEqual(base_image.height, self.test_new_small_image_height)
        self.assertEqual(base_image.width, self.test_new_small_image_width)

    def tearDown(self) -> None:
        return


if __name__ == "__main__":
    unittest.main()
