import os
from datetime import datetime

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

TEMP_DIR_NAME = "tmp"


def load_images(path):
    images = []
    valid_images = [".png"]
    for f in os.listdir(path):
        ext = os.path.splitext(f)[1]
        if ext.lower() not in valid_images:
            continue
        images.append(os.path.join(path, f))
    return images


def get_date_and_time():
    now = datetime.now()
    return now.strftime("%H_%M_%S__%d_%m_%Y")
