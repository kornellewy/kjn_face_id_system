import unittest
from pathlib import Path
import shutil

import numpy as np

from kjn_face_id_system.images.image_id_case import IdCaseKFISImage
from kjn_face_id_system.utils.utils import (
    create_single_base_example,
    IMAGE_TAG_VALUE_FOR_SELFIE,
)


class IdCaseKFISImageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.test_case_path = create_single_base_example()
        self.test_image_name = "id.png"
        self.test_image_path = self.test_case_path.joinpath(
            IMAGE_TAG_VALUE_FOR_SELFIE, self.test_image_name
        )

    def test_detect(self):
        detector = IdCaseKFISImage(
            case_path=self.test_case_path, image_name=self.test_image_name
        )
        detector.get_idcards()
        print(detector.bboxes)

    # TODO: Test image save update

    def tearDown(self) -> None:
        shutil.rmtree(self.test_case_path)


if __name__ == "__main__":
    unittest.main()
