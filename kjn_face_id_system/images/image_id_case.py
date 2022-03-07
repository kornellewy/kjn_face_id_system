from pathlib import Path
from typing import Optional, List

import cv2
from matplotlib.transforms import Bbox
import numpy as np
import torch

from kjn_face_id_system.images.bbox import BBox
from kjn_face_id_system.id_card_localization.id_card_localizator import (
    IdCardLocalizator,
)
from kjn_face_id_system.face_localization.face_localizator import (
    FaceLocalizator,
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
    IMAGE_NAME,
)


class IdCaseKFISImage:
    def __init__(
        self,
        case_path: Path,
        image_name: str,
        device: Optional[torch.device] = torch.device("cpu"),
    ) -> None:
        self.case_path = case_path
        self.image_name = image_name
        self.device = device
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

        self.id_card_localizator = IdCardLocalizator(device=device)
        self.face_localizator = FaceLocalizator(device=device)

    def check_case(self) -> None:
        if not self.id_card_front_path.exists():
            raise ValueError(
                f"id_card_front_path: {self.id_card_front_path} dont exist."
            )
        elif not self.id_card_back_path.exists():
            raise ValueError(f"id_card_back_path: {self.id_card_back_path} dont exist.")
        elif not self.id_selfie_path.exists():
            raise ValueError(f"id_selfie_path: {self.id_selfie_path} dont exist.")

    def get_faces(self) -> list:
        for image_path, image_tags in zip(
            self.start_images_paths, self.start_images_tags
        ):
            faces_bboxes = self.find_faces(image_path=image_path, image_tags=image_tags)
            faces_images = self.cut_faces(faces_bboxes=faces_bboxes)
            face_objects = self.create_faces_images_obejcts(
                faces_images=faces_images, image_tags=image_tags
            )
            self.face_images += face_objects
        return self.face_images

    def find_faces(self, image_path: Path, image_tags: dict) -> BBox:
        faces_bboxes = self.face_localizator.detect(image_path=image_path)
        for face_bbox in faces_bboxes:
            face_bbox.tags = image_tags
        self.bboxes += faces_bboxes
        return faces_bboxes

    def cut_faces(self, faces_bboxes: List[Bbox]) -> List[np.ndarray]:
        faces_images = []
        for face_bbox in faces_bboxes:
            full_image = BaseKFISImage(face_bbox.image_path)
            full_height, full_width = full_image.height, full_image.width
            bbox_as_coco = face_bbox.get_coco()
            bbox_height = int(bbox_as_coco["height"] * full_height)
            bbox_width = int(bbox_as_coco["width"] * full_width)
            bbox_x_top_left = int(bbox_as_coco["x_top_left"] * full_width)
            bbox_y_top_left = int(bbox_as_coco["y_top_left"] * full_height)
            crop_img = full_image.image[
                bbox_y_top_left : bbox_y_top_left + bbox_height,
                bbox_x_top_left : bbox_x_top_left + bbox_width,
            ]
            faces_images.append(crop_img)
        return faces_images

    def create_faces_images_obejcts(
        self, faces_images: List[np.ndarray], image_tags: dict
    ):
        self.faces_dir_path.mkdir(exist_ok=True, parents=True)
        face_objects = []
        for face_idx, face_image in enumerate(faces_images):
            face_image_path = self.faces_dir_path.joinpath(
                f"{image_tags[IMAGE_TAG_KEY_FOR_SOURCE]}_{face_idx}_{IMAGE_NAME}"
            )
            cv2.imwrite(face_image_path.as_posix(), face_image)
            face_image = BaseKFISImage(image_path=face_image_path, tags=image_tags)
            face_objects.append(face_image)
        return face_objects

    def get_idcards(self):
        for image_path, image_tags in zip(
            self.start_images_paths, self.start_images_tags
        ):
            id_card_bbox = self.find_id_card(
                image_path=image_path, image_tags=image_tags
            )
            id_card_image = self.cut_if_id_card(bbox=id_card_bbox)
            id_card_object = self.create_id_card_obejct(
                id_card_image=id_card_image, image_tags=image_tags
            )
            self.id_card_images.append(id_card_object)
        return self.id_card_images

    def find_id_card(self, image_path: Path, image_tags: dict) -> BBox:
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

    def create_id_card_obejct(self, id_card_image: List[np.ndarray], image_tags: dict):
        self.cut_id_cards_path.mkdir(exist_ok=True, parents=True)
        idx_card_image_path = self.cut_id_cards_path.joinpath(
            f"{image_tags[IMAGE_TAG_KEY_FOR_SOURCE]}_{IMAGE_NAME}"
        )
        cv2.imwrite(idx_card_image_path.as_posix(), id_card_image)
        id_card_image = BaseKFISImage(image_path=idx_card_image_path, tags=image_tags)
        return id_card_image

    def straightening_of_id_card(self) -> None:
        # prostowanie zdjencia bboxa
        # nie wiem czy bd potrzebne
        pass

    def classification_id_card_type(self) -> None:
        # klasyfikacja one shot do typu dowodu ktury mamy w bazie danych
        pass

    def final_case(self) -> None:
        # funkcja ktora sprawdza porpawnosć stowzeonego obiektu,
        # jego strukute, odczytane wartosci
        pass
