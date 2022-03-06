import os
from datetime import datetime
from pathlib import Path
import shutil

ID_CARD_BACK_DIR_NAME = "back_id"
ID_CARD_FRONT_DIR_NAME = "front_id"
SELFIE_DIR_NAME = "selfie"
FACES_DIR_NAME = "faces"
CUT_ID_CARDS_DIR_NAME = "cut_id_cards"

IMAGE_TAG_KEY_FOR_SOURCE = "source"
IMAGE_TAG_KEY_FOR_DATE = "date"
IMAGE_TAG_VALUE_FOR_SELFIE = "selfie"
IMAGE_TAG_VALUE_FOR_FRONT_ID = "front_id"
IMAGE_TAG_VALUE_FOR_BACK_ID = "back_id"
IMAGE_NAME = "id.png"

TEMP_DIR_NAME = "tmp"
TEST_CASE_EXAMPLE = "datasets/combinations/18"

ID_CARD_CLASS_NAME = "id_card"
FACE_CLASS_NAME = "face"


def load_images(path: str, valid_images: list = [".png"]) -> list:
    images = []
    for f in os.listdir(path):
        ext = os.path.splitext(f)[1]
        if ext.lower() not in valid_images:
            continue
        images.append(os.path.join(path, f))
    return images


def get_date_and_time() -> str:
    now = datetime.now()
    return now.strftime("%H_%M_%S__%d_%m_%Y")


def create_single_base_example() -> Path:
    source_example = TEST_CASE_EXAMPLE
    targer_ame = f"{source_example}_{get_date_and_time()}"
    target_path = Path(TEMP_DIR_NAME)
    target_path.mkdir(parents=True, exist_ok=True)
    target_path = target_path.joinpath(targer_ame)
    shutil.copytree(source_example, target_path, dirs_exist_ok=True, symlinks=True)
    return target_path


def convert_images_with_dir_to_png(dir_path: str) -> None:
    input_images_paths = load_images(dir_path, valid_images=[".png", ".jpg", ".jpeg"])
    for image_path in input_images_paths:
        new_image_path = Path(image_path).parent.joinpath(
            f"{Path(image_path).stem}.png"
        )
        shutil.copy2(image_path, new_image_path)
        os.remove(image_path)
        