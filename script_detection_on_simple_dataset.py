import glob
import os
import cv2

import hyperparams
from image import Image
from pipeline import Pipeline

pipeline = Pipeline()
total = 0
success = 0

for directory in os.listdir(hyperparams.simple_images_path):
    images = glob.glob(f'{hyperparams.simple_images_path}/{directory}/*.jpg')

    for image_path in images[:5]:
        image_name = os.path.basename(image_path)
        image = Image.load(image_path, hyperparams.image_width)

        if image is None:
            continue

        boxes, labels, scores = pipeline.run(image)
        image = Image.show_boxes_with_labels(image, boxes, labels, scores, False)
        cv2.imwrite(f'result/{image_name}', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        # cv2.waitKey()

        for i in range(len(boxes)):
            total += 1
            success += 1 if labels[i] == directory else 0

print(f'total: {total}')
print(success / total)
