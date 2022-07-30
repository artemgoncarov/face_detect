import dlib

from classifier import Classifier
from detector import Detector
from encoder import Encoder
from shape_predictor import ShapePredictor


class Pipeline:
    def __init__(self):
        self.detector = Detector()
        self.shape_predictor_68 = ShapePredictor()
        self.encoder = Encoder()
        self.classifier = Classifier()

    def run(self, image):
        boxes = self.detector.detect(image)
        labels, scores = [], []

        for box in boxes:
            landmarks = self.shape_predictor_68.get_landmarks(image, box)
            face_chip = dlib.get_face_chip(image, landmarks)
            encoding = self.encoder.encode(face_chip)
            label, score = self.classifier.classify(encoding)
            labels.append(label)
            scores.append(score)

        return boxes, labels, scores
