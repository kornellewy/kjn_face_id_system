import unittest
from pathlib import Path

import numpy as np

from kjn_face_id_system.images.id_card import KFISIdCase

from kjn_face_id_system.utils.utils import (
    IMAGE_TAG_KEY_FOR_SOURCE,
    IMAGE_TAG_KEY_FOR_DATE,
    IMAGE_TAG_VALUE_FOR_FRONT_ID,
    get_date_and_time,
)


class KFISIdCaseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.test_image_path = Path("test_case_example/front_id/id.png")
        self.test_image_height = 768
        self.test_image_width = 1024
        self.case_path = Path("test_case_example")
        self.tags = (
            {
                IMAGE_TAG_KEY_FOR_SOURCE: IMAGE_TAG_VALUE_FOR_FRONT_ID,
                IMAGE_TAG_KEY_FOR_DATE: get_date_and_time(),
            },
        )

    def test_creat_image(self):
        base_image = KFISIdCase(
            image_path=self.test_image_path, case_path=self.case_path, tags=self.tags
        )
        self.assertIsInstance(base_image.image, np.ndarray)
        self.assertEqual(base_image.height, self.test_image_height)
        self.assertEqual(base_image.width, self.test_image_width)

    def tearDown(self) -> None:
        return


if __name__ == "__main__":
    unittest.main()
