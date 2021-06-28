import tensorflow as tf
from unetModel import unetModel
from plotImage import display
from cocoDataset import CocoDataset
from pycocotools import mask as maskUtils
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, ModelCheckpoint


IMAGE_SIZE = 448

# Hyperparameters
BATCH_SIZE = 8
EPOCHS = 50

def annToRLE(ann, height, width):
    """
    Convert annotation which can be polygons, uncompressed RLE to RLE.
    :return: binary mask (numpy 2D array)
    """
    segm = ann["segmentation"]
    if isinstance(segm, list):
        # polygon -- a single object might consist of multiple parts
        # we merge all parts into one mask rle code
        if not isinstance(segm[0], list):
            segm = [segm]
        rles = maskUtils.frPyObjects(segm, height, width)
        rle = maskUtils.merge(rles)
    elif isinstance(segm["counts"], list):
        # uncompressed RLE
        rle = maskUtils.frPyObjects(segm, height, width)
    else:
        # rle
        rle = ann["segmentation"]
    return rle

def buildUNetMask(cocoItem):
    (image, _, annotations) = cocoItem
    
    annMasks = [[np.zeros((IMAGE_SIZE, IMAGE_SIZE))], [np.zeros((IMAGE_SIZE, IMAGE_SIZE))], [np.zeros((IMAGE_SIZE, IMAGE_SIZE))], [np.zeros((IMAGE_SIZE, IMAGE_SIZE))]]
    for annotation in annotations:
        annMask = maskUtils.decode(annToRLE(annotation, IMAGE_SIZE, IMAGE_SIZE))
        classNumber = int(annotation["category_id"])
        annMasks[classNumber].append(annMask)
    masks = np.concatenate([np.expand_dims(np.maximum.reduce(annClassMasks), axis=2) for annClassMasks in annMasks], axis=2)
    return image, masks

def resizeToUNet(image, mask):
    return (
        tf.constant(cv.resize(image, (572, 572))),
        tf.constant(cv.resize(mask, (388, 388)))
    )

@tf.function
def load_image_train(image, mask):
    # Randomly choosing the images to flip right left.
    # We need to split both the input image and the input mask as the mask is in correspondence to the input image.
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
        mask = tf.image.flip_left_right(mask)

    # Normalizing the input image.
    image = tf.cast(image, tf.float32) / 255.0

    # Returning the input_image and the input_mask
    return image, mask

# We do not need to do any image augmentation here for the validation dataset.
def load_image_test(image, mask):
    # Normalizing the input image.
    image = tf.cast(image, tf.float32) / 255.0

    return image, mask

def main(model_path: str):
    if not model_path:
        test_dataset = CocoDataset(root="./data_RGB/test", annFile="./data_RGB/test/test.json")
        test_dataset = [buildUNetMask(item) for item in test_dataset]
        test_dataset = [resizeToUNet(image, mask) for image, mask in test_dataset]
        test_dataset = tf.data.Dataset.from_tensor_slices(([x[0] for x in test_dataset], [x[1] for x in test_dataset]))
        test_dataset = test_dataset.map(load_image_test)
        test_dataset = test_dataset.batch(BATCH_SIZE)

        train_dataset = CocoDataset(root="./data_RGB/train", annFile="./data_RGB/train/train.json")
        train_dataset = [buildUNetMask(item) for item in train_dataset]
        train_dataset = [resizeToUNet(image, mask) for image, mask in train_dataset]
        train_dataset = tf.data.Dataset.from_tensor_slices(([x[0] for x in train_dataset], [x[1] for x in train_dataset]))
        train_dataset = train_dataset.map(load_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        train_dataset = train_dataset.batch(BATCH_SIZE)

        # Create the model.
        model = unetModel()
        # Build the model with the input shape
        # Image is RGB, so here the input channel is 3.
        model.build(input_shape=(None, 3, 572, 572))
        model.summary()

        earlystopper = EarlyStopping(patience=5, verbose=1)

        checkpoint = ModelCheckpoint( '/content/gdrive/My Drive/iranian_models/unet/model.h5', monitor='val_loss', verbose=1, 
                                    save_best_only=True, mode='min')

        model.compile(optimizer='adam',
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics=['accuracy'])
        model_history = model.fit(train_dataset, epochs=EPOCHS, validation_data=test_dataset, callbacks=[earlystopper, checkpoint])

    if model_path:
        test_dataset_raw = CocoDataset(root="./data_RGB/test", annFile="./data_RGB/test/test.json")
        model = tf.keras.models.load_model(model_path)
        for item in list(test_dataset_raw)[0:3]:
            test_dataset = [buildUNetMask(item)]
            test_dataset = [resizeToUNet(image, mask) for image, mask in test_dataset]
            test_dataset = tf.data.Dataset.from_tensor_slices(([x[0] for x in test_dataset], [x[1] for x in test_dataset]))
            test_dataset = test_dataset.map(load_image_test)
            test_dataset = test_dataset.batch(1)
            xd = model.predict(test_dataset)[0]
            plt.figure()
            plt.imshow(item[0], 'gray', interpolation='none')
            plt.show()
            plt.figure()
            plt.imshow(xd[:, :, 0], 'gray', interpolation='none')
            plt.show()
            plt.figure()
            plt.imshow(xd[:, :, 1], 'gray', interpolation='none')
            plt.show()
            plt.figure()
            plt.imshow(xd[:, :, 2], 'gray', interpolation='none')
            plt.show()
            plt.figure()
            plt.imshow(xd[:, :, 3], 'gray', interpolation='none')
            plt.show()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description=__doc__)
    parser.add_argument('--model-path', help='model_path')
    args = parser.parse_args()
    main(args.model_path)
