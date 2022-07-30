import dlib


class Detector:
    def __init__(self):
        self.model = dlib.get_frontal_face_detector()

    def detect(self, image):
        return self.model(image, 1)
