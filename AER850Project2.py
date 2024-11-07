import keras
from keras.preprocessing import image_dataset_from_directory
from keras.layers import Rescaling, RandomZoom, RandomTranslation, RandomFlip

trainset = image_dataset_from_directory(
    'data/train',
    image_size=(500, 500),
    batch_size=32,
    label_mode='categorical'
)

validset = image_dataset_from_directory(
    'data/valid',
    image_size=(500, 500),
    batch_size=32,
    label_mode='categorical'
)

augment = keras.Sequential([
    Rescaling(1./255),
    RandomZoom(0.1),
    RandomTranslation(0.1,0.1),
    RandomFlip("horizontal")
])

trainset = trainset.map(lambda x, y: (augment(x, training=True), y))

validset = validset.map(lambda x, y: (Rescaling(1./255)(x), y))
