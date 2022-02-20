import unittest
from pathlib import Path

import numpy as np

from kjn_face_id_system.id_card_localization.id_card_localizator import (
    IdCardLocalizator,
)


class IdCardLocalizatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.test_image_path = Path("test_case_example/selfie/id.png")

    def test_creat_localizator(self):
        localizator = IdCardLocalizator()
        bbox = localizator.detect(image_path=self.test_image_path)
        print(bbox)

    def test_detect(self):
        localizator = IdCardLocalizator()
        bbox = localizator.detect(image_path=self.test_image_path)
        self.assertEqual(len(bbox), 6)


if __name__ == "__main__":
    unittest.main()
