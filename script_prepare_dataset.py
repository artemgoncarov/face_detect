import os
import glob
import json
import numpy as np


import hyperparams
from encoder import Encoder
from image import Image


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


encoder = Encoder()

os.makedirs(hyperparams.classifier_dataset_json_path, exist_ok=True)

for folder in os.listdir(hyperparams.classifier_dataset_image_path):
    image_dir_path = f'{hyperparams.classifier_dataset_image_path}/{folder}'
    json_dir_path = f'{hyperparams.classifier_dataset_json_path}/{folder}'

    images = glob.glob(f'{image_dir_path}/*.jpg')

    os.makedirs(json_dir_path, exist_ok=True)

    for image_path in images:
        image_filename = os.path.splitext(os.path.basename(image_path))[0]
        image = Image.load(image_path, (hyperparams.face_image_width, hyperparams.face_image_height))

        if image is None:
            continue

        features = encoder.encode(image)
        print(features.shape)

        if len(features) > 0:
            with open(f'{json_dir_path}/{image_filename}.json', 'w') as file:
                file.write(json.dumps(features, cls=NumpyEncoder))
