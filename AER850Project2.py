import keras
from keras import layers, models
from keras.preprocessing import image_dataset_from_directory
from keras.optimizers import Adam
import matplotlib.pyplot as plt

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
    layers.RandomRotation()
])

trainset = trainset.map(lambda x, y: (augment(x, training=True), y))

validset = validset.map(lambda x, y: (layers.Rescaling(1./255)(x), y))

""" Step 2 & 3: Neural Network Architecture Design & Hyperparameter Analysis"""

mymodel = models.Sequential([

    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(500, 500, 3)),
    layers.MaxPooling2D((2, 2)),


    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),


    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(256, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(), # turns 2D into 1D data
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3), # drop some of the neurons
    layers.Dense(3, activation='softmax')]
)

# configer the model with loss and metrics

mymodel.compile(optimizer=Adam(learning_rate=0.01),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# summary of the model showing dimensions and # of parameters for each layer
mymodel.summary()

""" Step 4: Model Evaluation """

fittedmodel = mymodel.fit(
    trainset,
    validation_data=validset,
    epochs = 2
)

fig, axes = plt.subplots(1, 2)

# Access history of the fitted model
axes[0].plot(fittedmodel.history['accuracy'], label='Training Accuracy')
axes[0].plot(fittedmodel.history['val_accuracy'], label='Validation Accuracy')
axes[0].set_title('Model Accuracy')
axes[0].legend()

axes[1].plot(fittedmodel.history['loss'], label='Training Loss')
axes[1].plot(fittedmodel.history['val_loss'], label='Validation Loss')
axes[1].set_title('Model Loss')
axes[1].legend()

plt.show()