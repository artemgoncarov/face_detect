import cv2

import hyperparams
from image import Image
from pipeline import Pipeline

pipeline = Pipeline()
image_path = 'data/group_face_dataset/IKhajcTRV7Q.jpg'
image = Image.load(image_path, hyperparams.image_width)

boxes, labels, scores = pipeline.run(image)
image = Image.show_boxes_with_labels(image, boxes, labels, scores, False)
cv2.imwrite('output.jpg', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
