import keras
from keras import layers, models
from keras.preprocessing import image_dataset_from_directory

""" Step 1: Data Processing """

# Generators

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

# Data Augmentation

augment = models.Sequential([
    layers.Rescaling(1./255),
    layers.RandomZoom(0.1),
    layers.RandomTranslation(0.1,0.1),
    layers.RandomFlip("horizontal")
])

trainset = trainset.map(lambda x, y: (augment(x, training=True), y))

validset = validset.map(lambda x, y: (layers.Rescaling(1./255)(x), y))

""" Step 2: Neural Network Architecture Design """

mymodel = models.Sequential([

    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(500, 500, 3)),
    layers.MaxPooling2D((2, 2)),


    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),


    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),


    layers.Flatten(), # turns 2D into 1D data
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5), # drop half of the neurons
    layers.Dense(3, activation='relu')]
)