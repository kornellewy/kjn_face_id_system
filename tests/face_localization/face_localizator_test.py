import unittest
import shutil

from kjn_face_id_system.images.bbox import BBox
from kjn_face_id_system.face_localization.face_localizator import FaceLocalizator
from kjn_face_id_system.utils.utils import (
    create_single_base_example,
    ID_CARD_FRONT_DIR_NAME,
    FACE_CLASS_NAME,
)


class FaceLocalizatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.test_case_path = create_single_base_example()
        self.test_image_name = "id.png"
        self.test_image_path = self.test_case_path.joinpath(
            ID_CARD_FRONT_DIR_NAME, self.test_image_name
        )

    def test_detect(self):
        detector = FaceLocalizator()
        bboxes = detector.detect(self.test_image_path)
        self.assertTrue(isinstance(bboxes, list))
        bbox = bboxes[0]
        self.assertTrue(isinstance(bbox, BBox))
        self.assertTrue(bbox.bbox_class == FACE_CLASS_NAME)
        self.assertTrue(bbox.x_center > 0)
        self.assertTrue(bbox.x_center < 1)
        self.assertTrue(bbox.y_center > 0)
        self.assertTrue(bbox.y_center < 1)
        self.assertTrue(bbox.width > 0)
        self.assertTrue(bbox.width < 1)
        self.assertTrue(bbox.height > 0)
        self.assertTrue(bbox.height < 1)
        self.assertTrue(bbox.confidence > 0)
        self.assertTrue(bbox.confidence < 1)

    def tearDown(self) -> None:
        shutil.rmtree(self.test_case_path)


if __name__ == "__main__":
    unittest.main()
