import os
import glob


import hyperparams


total = 0

for directory in os.listdir(hyperparams.classifier_dataset_json_path):
    images = glob.glob(f'{hyperparams.classifier_dataset_json_path}/{directory}/*.json')
    print(f'{directory}: {len(images)}')
    total += len(images)

print(f'total: {total}')
