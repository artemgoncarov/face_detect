import dlib
import numpy as np

import hyperparams


class Encoder:
    def __init__(self, path=hyperparams.model_face_recognition_path):
        self.model = dlib.face_recognition_model_v1(path)

    def encode(self, face_chip):
        features = self.model.compute_face_descriptor(face_chip)
        return np.array(features)
