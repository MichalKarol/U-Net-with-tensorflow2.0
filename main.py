import tensorflow as tf
from unetModel import unetModel
from plotImage import display
from cocoDataset import CocoDataset
from pycocotools import mask as maskUtils
import numpy as np
import cv2 as cv

IMAGE_SIZE = 448

# Hyperparameters
BATCH_SIZE = 8
BUFFER_SIZE = 1000
EPOCHS = 20

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
    
    annMasks = []
    for annotation in annotations:
        annMask = maskUtils.decode(annToRLE(annotation, IMAGE_SIZE, IMAGE_SIZE))
        annMask = annMask * int(annotation["category_id"])
        annMasks.append(annMask)
    return image, np.expand_dims(np.maximum.reduce(annMasks), axis=2)

def resizeToUNet(image, mask):
    return (
        cv.resize(image, (572, 572)),
        np.expand_dims(cv.resize(mask, (388, 388)), axis=2)
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


train_dataset_generator = CocoDataset(root="./data_RGB/train", annFile="./data_RGB/train/train.json")
train_dataset_generator = (buildUNetMask(item) for item in train_dataset_generator)
train_dataset_generator = (resizeToUNet(image, mask) for image, mask in train_dataset_generator)
train_dataset = tf.data.Dataset.from_generator(
    lambda: train_dataset_generator,
    output_signature=(
          tf.TensorSpec(shape=(572, 572, 3), dtype=tf.int32),
          tf.TensorSpec(shape=(388, 388, 1), dtype=tf.int32)
    )
)

train_dataset = train_dataset.map(load_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
train_dataset = train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).cache()
train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

test_dataset_generator = CocoDataset(root="./data_RGB/test", annFile="./data_RGB/test/test.json")
test_dataset_generator = (buildUNetMask(item) for item in test_dataset_generator)
test_dataset_generator = (resizeToUNet(image, mask) for image, mask in test_dataset_generator)
test_dataset = tf.data.Dataset.from_generator(
    lambda: test_dataset_generator,
    output_signature=(
          tf.TensorSpec(shape=(572, 572, 3), dtype=tf.int32),
          tf.TensorSpec(shape=(388, 388, 1), dtype=tf.int32)
    )
)
test_dataset = test_dataset.map(load_image_test)
test_dataset = test_dataset.batch(BATCH_SIZE)


# Create the model.
model = unetModel()
# Build the model with the input shape
# Image is RGB, so here the input channel is 3.
model.build(input_shape=(None, 3, 572, 572))
model.summary()

# Write model saving callback.
model_save_callback = tf.keras.callbacks.ModelCheckpoint(
    '/content/gdrive/My Drive/iranian_models/unet/model_checkpoint', monitor='val_loss', verbose=0, save_best_only=True,
    save_weights_only=False, mode='auto', save_freq='epoch')

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model_history = model.fit(train_dataset, epochs=EPOCHS, validation_data=test_dataset, callbacks=[model_save_callback])
