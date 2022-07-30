import dlib

import hyperparams


class ShapePredictor:
    def __init__(self, path=hyperparams.model_shape_predictor_68_path):
        self.model = dlib.shape_predictor(path)

    def get_landmarks(self, image, box):
        landmarks = self.model(image, box)
        return landmarks
