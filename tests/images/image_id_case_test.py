import unittest
from pathlib import Path
import shutil

import numpy as np

from kjn_face_id_system.images.image_id_case import IdCaseKFISImage
from kjn_face_id_system.utils.utils import (
    create_single_base_example,
    IMAGE_TAG_VALUE_FOR_SELFIE,
    IMAGE_NAME,
    TEMP_DIR_NAME,
)


class IdCaseKFISImageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.test_case_path = create_single_base_example()
        self.test_image_name = IMAGE_NAME
        self.test_image_path = self.test_case_path.joinpath(
            IMAGE_TAG_VALUE_FOR_SELFIE, self.test_image_name
        )

    def test_get_idcards(self):
        detector = IdCaseKFISImage(
            case_path=self.test_case_path, image_name=self.test_image_name
        )
        detector.get_idcards()
        self.assertAlmostEqual(len(detector.bboxes), 3)
        self.assertAlmostEqual(len(detector.id_card_images), 3)

    def test_get_faces(self):
        detector = IdCaseKFISImage(
            case_path=self.test_case_path, image_name=self.test_image_name
        )
        detector.get_faces()
        self.assertAlmostEqual(len(detector.bboxes), 6)
        self.assertAlmostEqual(len(detector.face_images), 6)

    def test_get_faces_and_get_idcards(self):
        detector = IdCaseKFISImage(
            case_path=self.test_case_path, image_name=self.test_image_name
        )
        detector.get_idcards()
        detector.get_faces()
        self.assertAlmostEqual(len(detector.bboxes), 9)

    def tearDown(self) -> None:
        shutil.rmtree(TEMP_DIR_NAME)


if __name__ == "__main__":
    unittest.main()
