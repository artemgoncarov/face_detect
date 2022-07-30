import cv2


from image import Image
from pipeline import Pipeline


camera = cv2.VideoCapture(0)
pipeline = Pipeline()

for i in range(30):
    camera.read()

while True:
    ret, frame = camera.read()

    if not ret:
        continue

    image = Image.preprocess(frame)
    boxes, labels, scores = pipeline.run(image)
    Image.show_boxes_with_labels(image, boxes, labels, scores, True)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
