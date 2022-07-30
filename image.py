import itertools
import cv2


import hyperparams


class Image:
    @staticmethod
    def load(path, image_size=(hyperparams.image_width, hyperparams.image_height)):
        image = cv2.imread(path)
        if image is None:
            return None

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if isinstance(image_size, int):
            ratio = image.shape[0] / image.shape[1]
            image_height = int(image_size * ratio)

            image = cv2.resize(image, (image_size, image_height))
        else:
            image = cv2.resize(image, image_size)

        return image

    @staticmethod
    def preprocess(image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ratio = image.shape[0] / image.shape[1]
        image_height = int(hyperparams.image_width * ratio)
        image = cv2.resize(image, (hyperparams.image_width, image_height))

        return image

    @staticmethod
    def show(image):
        cv2.imshow("Image", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    @staticmethod
    def show_boxes(image, boxes):
        show_image = image.copy()

        for box in boxes:
            sx = max(0, box.left())
            sy = max(0, box.top())
            ex = min(box.right(), image.shape[1])
            ey = min(box.bottom(), image.shape[0])
            cv2.rectangle(show_image, (sx, sy), (ex, ey), (255, 0, 0), 5)

        Image.show(show_image)

    @staticmethod
    def show_boxes_with_labels(image, boxes, labels, scores, show=True):
        show_image = image.copy()

        for (box, label, score) in itertools.zip_longest(boxes, labels, scores):
            sx = max(0, box.left())
            sy = max(0, box.top())
            ex = min(box.right(), image.shape[1])
            ey = min(box.bottom(), image.shape[0])
            cv2.rectangle(show_image, (sx, sy), (ex, ey), (255, 0, 0), 5)

            cv2.putText(show_image,
                        f"{label} {round(100 * score, 1)}%",
                        (sx, sy-5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 0, 0),
                        1,
                        cv2.LINE_AA)

        if show is True:
            Image.show(show_image)

        return show_image
