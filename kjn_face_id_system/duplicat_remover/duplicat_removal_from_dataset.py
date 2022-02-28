from PIL import Image
import imagehash
import os


def yield_files_with_extensions(folder_path, file_extension):
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(file_extension):
                yield os.path.join(root,file)

def load_images2(path):
    images_exts = ('.jpg', '.jpeg', '.png')
    return list(yield_files_with_extensions(path, images_exts))

class DuplicatImageRemover(object):
    def __init__(self):
        super().__init__()

    def remove_duplicats(self, dataset_path):
        return self._remove_duplicats(dataset_path)

    def _remove_duplicats(self, dataset_path):
        images_paths = self._get_all_images_paths_in_dataset(dataset_path)
        img_hashes = {}
        [self._img_hash_calculate_remove(img_path, img_hashes) for img_path in images_paths]

    def _get_all_images_paths_in_dataset(self, dataset_path):
        class_paths = [f.path for f in os.scandir(dataset_path) if f.is_dir()]
        images_paths = []
        if class_paths:
            for class_path in class_paths:
                images_paths += load_images2(class_path)
        else:
            images_paths += load_images2(dataset_path)
        return images_paths

    def _img_hash_calculate_remove(self, img_path, img_hashes):
        img_hash = imagehash.average_hash(Image.open(img_path))
        if img_hash in img_hashes:
            # print( '{} duplicate of {}'.format(img_path, img_hashes[img_hash]) )
            os.remove(img_path)
        else:
            img_hashes[img_hash] = img_path

    def remove_near_duplicats(self, dataset_path, treshhold_epsilon=50):
        return self._remove_near_duplicats(dataset_path, treshhold_epsilon)

    def _remove_near_duplicats(self, dataset_path, treshhold_epsilon):
        images_paths = self._get_all_images_paths_in_dataset(dataset_path)
        [ self._img_near_hash_calculate_remove(img_path1, img_path2, treshhold_epsilon) for img_path1, img_path2 in zip(images_paths, images_paths[::-1]) if img_path1 != img_path2 ]
            
    def _img_near_hash_calculate_remove(self, img_path1, img_path2, treshhold_epsilon):
        hash1 = imagehash.average_hash(Image.open(img_path1))
        hash2 = imagehash.average_hash(Image.open(img_path2))
        if hash1 - hash2 < treshhold_epsilon:
            os.remove(img_path1)
            os.remove(img_path2)
            # print( '{} is near duplicate of {}'.format(img_path1, img_path2) )

if __name__ == '__main__':
    data_set_path = 'J:/web_scraper/images'
    kjn = DuplicatImageRemover()
    kjn.remove_duplicats(data_set_path)
    