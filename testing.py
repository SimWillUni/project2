import numpy as np
from keras.preprocessing import image
from keras.models import load_model
import matplotlib.pyplot as plt

mymodel = load_model("../mymodel.keras")
classes = ["Crack", "Missing Head", "Paint Off"]

# func to predict the class and display the image
def display_prediction(img_path):
    img = image.load_img(img_path, target_size=(500, 500))
    img_array = image.img_to_array(img) / 255.0  # normalizing the img data
    img_array = np.expand_dims(img_array, axis=0)

    # class prediction
    predictions = mymodel.predict(img_array)[0]
    predicted_class = classes[np.argmax(predictions)]
    
    plt.imshow(img)
    plt.axis('off')
    
    # putting the text on the image
    text = "\n".join([f"{classes[i]}: {predictions[i]:.2f}" for i in range(len(classes))])
    plt.text(
        10, 450, text, color="green", fontsize=16, 
        bbox=dict(facecolor="white", alpha=0.5)
    )
    plt.title(f"Predicted: {predicted_class}")
    plt.show()

display_prediction("data/test/crack/test_crack.jpg")
display_prediction("data/test/paint-off/test_paintoff.jpg")
display_prediction("data/test/missing-head/test_missinghead.jpg")