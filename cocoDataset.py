import tensorflow as tf
import cv2 as cv
from pycocotools.coco import COCO
import os

def CocoDataset(root: str, annFile: str):
    coco = COCO(annFile)
    ids = list(sorted(coco.imgs.keys()))

    for imageid in ids:
        image_path = os.path.join(root, coco.loadImgs(imageid)[0]["file_name"])
        image = cv.imread(image_path)
        annotations = coco.loadAnns(coco.getAnnIds(imageid))
        yield (image, image_path, annotations)
