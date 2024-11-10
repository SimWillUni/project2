import numpy as np
from keras.preprocessing import image
from keras.models import load_model
import matplotlib.pyplot as plt

mymodel = load_model("../mymodel.keras")
classes = ["Crack", "Missing Head", "Paint Off"]

""" test image paths """

#data/test/crack/test_crack.jpg
#data/test/crack/test_paintoff.jpg
#data/test/crack/test_missinghead.jpg

