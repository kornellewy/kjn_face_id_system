from pathlib import Path
from typing import Optional, List

import cv2
from matplotlib.transforms import Bbox
import numpy as np

from kjn_face_id_system.images.bbox import BBox
from kjn_face_id_system.id_card_localization.id_card_localizator import (
    IdCardLocalizator,
)
from kjn_face_id_system.images.base_image import BaseKFISImage
from kjn_face_id_system.utils.utils import (
    ID_CARD_BACK_DIR_NAME,
    ID_CARD_FRONT_DIR_NAME,
    SELFIE_DIR_NAME,
    FACES_DIR_NAME,
    CUT_ID_CARDS_DIR_NAME,
    IMAGE_TAG_KEY_FOR_SOURCE,
    IMAGE_TAG_KEY_FOR_DATE,
    IMAGE_TAG_VALUE_FOR_SELFIE,
    IMAGE_TAG_VALUE_FOR_FRONT_ID,
    IMAGE_TAG_VALUE_FOR_BACK_ID,
    get_date_and_time,
)


class IdCaseKFISImage:
    def __init__(self, case_path: Path, image_name: str) -> None:
        self.case_path = case_path
        self.image_name = image_name
        self.id_card_front_dir_path = case_path.joinpath(ID_CARD_FRONT_DIR_NAME)
        self.id_card_front_path = self.id_card_front_dir_path.joinpath(self.image_name)
        self.id_card_back_dir_path = case_path.joinpath(ID_CARD_BACK_DIR_NAME)
        self.id_card_back_path = self.id_card_back_dir_path.joinpath(self.image_name)
        self.selfie_dir_path = case_path.joinpath(SELFIE_DIR_NAME)
        self.id_selfie_path = self.selfie_dir_path.joinpath(self.image_name)

        self.create_time = get_date_and_time()

        self.start_images_paths = [
            self.id_card_front_path,
            self.id_card_back_path,
            self.id_selfie_path,
        ]
        self.start_images_tags = [
            {
                IMAGE_TAG_KEY_FOR_SOURCE: IMAGE_TAG_VALUE_FOR_FRONT_ID,
                IMAGE_TAG_KEY_FOR_DATE: self.create_time,
            },
            {
                IMAGE_TAG_KEY_FOR_SOURCE: IMAGE_TAG_VALUE_FOR_BACK_ID,
                IMAGE_TAG_KEY_FOR_DATE: self.create_time,
            },
            {
                IMAGE_TAG_KEY_FOR_SOURCE: IMAGE_TAG_VALUE_FOR_SELFIE,
                IMAGE_TAG_KEY_FOR_DATE: self.create_time,
            },
        ]
        self.check_case()

        self.faces_dir_path = case_path.joinpath(FACES_DIR_NAME)
        self.cut_id_cards_path = case_path.joinpath(CUT_ID_CARDS_DIR_NAME)

        self.bboxes = []  # bboxes for face images and idcard
        self.face_images = []  # n images of face
        self.id_card_images = []  # place for idcard images

        self.id_card_localizator = IdCardLocalizator()

    def check_case(self) -> None:
        if not self.id_card_front_path.exists():
            raise ValueError(
                f"id_card_front_path: {self.id_card_front_path} dont exist."
            )
        elif not self.id_card_back_path.exists():
            raise ValueError(f"id_card_back_path: {self.id_card_back_path} dont exist.")
        elif not self.id_selfie_path.exists():
            raise ValueError(f"id_selfie_path: {self.id_selfie_path} dont exist.")

    def get_faces(self) -> None:
        # znajduje zdjecia twazy, towzy obiekty face image z tagami
        # w zaleznosci gdzie zdjecie sie znajduje
        pass

    def find_faces(self) -> Optional[List[np.ndarray]]:
        # znajduje zdjecia twazy na zdjeciu
        pass

    def get_paths_for_face_images(self) -> List[Path]:
        # daje sieszki do zapisj zdjec twazy
        pass

    def save_faces(
        self, face_images: List[np.ndarray], save_images_paths: List[Path]
    ) -> None:
        # zapisz twaze znalezione na 1 zdjeciu
        pass

    def get_idcards(self):
        # iterujesz po wszytkich bazowych zdjeciach szukajac zdjecia dowdu
        for image_path, image_tags in zip(
            self.start_images_paths, self.start_images_tags
        ):
            id_card_bbox = self.find_id_cards(
                image_path=image_path, image_tags=image_tags
            )
            crop_img = self.cut_if_id_card(bbox=id_card_bbox)
            cv2.imshow("crop_img", crop_img)
            cv2.waitKey(0)

    def find_id_cards(self, image_path: Path, image_tags: dict) -> BBox:
        id_card_bbox = self.id_card_localizator.detect(image_path)
        id_card_bbox.tags = image_tags
        self.bboxes.append(id_card_bbox)
        return id_card_bbox

    def cut_if_id_card(self, bbox: BBox) -> np.ndarray:
        full_image = BaseKFISImage(bbox.image_path)
        full_height, full_width = full_image.height, full_image.width
        bbox_as_coco = bbox.get_coco()
        bbox_height = int(bbox_as_coco["height"] * full_height)
        bbox_width = int(bbox_as_coco["width"] * full_width)
        bbox_x_top_left = int(bbox_as_coco["x_top_left"] * full_width)
        bbox_y_top_left = int(bbox_as_coco["y_top_left"] * full_height)
        crop_img = full_image.image[
            bbox_y_top_left : bbox_y_top_left + bbox_height,
            bbox_x_top_left : bbox_x_top_left + bbox_width,
        ]
        return crop_img

    def straightening_of_id_card(self, id_card: np.ndarray) -> Optional[np.ndarray]:
        # prostowanie zdjencia bboxa
        pass

    def create_id_card(self, id_card: np.ndarray) -> None:
        # twozymy objekt id card ktory czyhta karty i
        # napisy i zapisuje odczytane wartosic
        pass

    def final_case(self) -> None:
        # funkcja ktora sprawdza porpawnosć stowzeonego obiektu,
        # jego strukute, odczytane wartosci
        pass
