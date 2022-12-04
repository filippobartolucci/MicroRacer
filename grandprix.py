from datetime import datetime
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
from tensorflow.keras import regularizers
import numpy as np
import matplotlib.pyplot as plt
import tracks
import os



class Model:
    def __init__(self):
        return

    def load_weight(self, weights_file_actor):
        self.actor_model = keras.models.load_model(weights_file_actor)

    def get_actor_model(self):
        return self.actor_model

models = []

for file in os.listdir("weights"):

    if os.path.isdir(os.path.join("weights", file)) and "actor" in file:
        print("Loading model: " + file)
        model = Model()
        model.load_weight(os.path.join("weights", file))
        models.append((model.get_actor_model(), file))


tracks.newrun(models)
