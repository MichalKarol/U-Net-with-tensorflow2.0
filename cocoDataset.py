import tensorflow as tf
import cv2 as cv
from pycocotools.coco import COCO
import os

class CocoDataset:
    class CocoDatasetIterator:
        def __init__(self, dataset):
            self._dataset = dataset
            self._index = 0

        def __next__(self):
            if (self._index < len(self._dataset._coco)):
                imageid = self._dataset._coco[self._index]
                image_path = os.path.join(self._dataset._root, self._dataset._coco.loadImgs(imageid)[0]["file_name"])
                image = cv.imread(image_path)
                annotations = self._dataset._coco.loadAnns(self._dataset._coco.getAnnIds(imageid))
                self._index +=1
                return (image, image_path, annotations)
            raise StopIteration

    def __init__(self, root: str, annFile: str):
        self._coco = COCO(annFile)
        self._ids = list(sorted(self.coco.imgs.keys()))
        self._root = root

    def __iter__(self):
       return CocoDatasetIterator(self)
