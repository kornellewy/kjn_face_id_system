import unittest
import shutil

from kjn_face_id_system.images.bbox import BBox
from kjn_face_id_system.utils.utils import (
    create_single_base_example,
    IMAGE_TAG_VALUE_FOR_SELFIE,
    ID_CARD_CLASS_NAME,
)
from kjn_face_id_system.id_card_localization.id_card_localizator import (
    IdCardLocalizator,
)


class LocalizatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.test_case_path = create_single_base_example()
        self.test_image_name = "id.png"
        self.test_image_path = self.test_case_path.joinpath(
            IMAGE_TAG_VALUE_FOR_SELFIE, self.test_image_name
        )

    def test_detect(self):
        detector = IdCardLocalizator()
        bbox = detector.detect(self.test_image_path)
        self.assertTrue(isinstance(bbox, BBox))
        self.assertTrue(bbox.bbox_class == ID_CARD_CLASS_NAME)
        self.assertTrue(bbox.x_center > 0)
        self.assertTrue(bbox.y_center > 0)
        self.assertTrue(bbox.width > 0)
        self.assertTrue(bbox.height > 0)
        self.assertTrue(bbox.confidence > 0)

    def tearDown(self) -> None:
        shutil.rmtree(self.test_case_path)


if __name__ == "__main__":
    unittest.main()
