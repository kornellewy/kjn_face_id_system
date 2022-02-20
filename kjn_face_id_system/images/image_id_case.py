from pathlib import Path
from typing import Optional, List

import cv2
import numpy as np

from kjn_face_id_system.id_card_localization.detect import run as detect_id_card
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
            self.selfie_dir_path,
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
            pass

    def find_id_cards(self, id_card: np.ndarray) -> Optional[np.ndarray]:
        # znajduje dowody na zdjecnciu zapisuje jak wygladaja
        # i zapisuje txt z lokazlizacja i zwaca wycientego bbox idcard
        pass

    def straightening_of_id_card(self, id_card: np.ndarray) -> Optional[np.ndarray]:
        # prostowanie zdjencia bboxa
        pass

    def create_id_card(self, id_card: np.ndarray) -> :