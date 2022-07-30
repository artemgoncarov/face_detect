model_shape_predictor_5_path = 'data/dlib_models/shape_predictor_5_face_landmarks.dat'
model_shape_predictor_68_path = 'data/dlib_models/shape_predictor_68_face_landmarks.dat'
model_face_recognition_path = 'data/dlib_models/dlib_face_recognition_resnet_model_v1.dat'
model_face_classification_path = 'data/dlib_models/classifier/checkpoint'

simple_images_path = 'data/simple_face_dataset'

image_width = 1024
image_height = 1024

face_image_width = 150
face_image_height = 150

detector_upsample = 1

classifier_dataset_image_path = 'data/cut_face_dataset'
classifier_dataset_json_path = 'data/json_face_dataset'

classifier_batch_size = 32
classifier_val_split = .1
classifier_epochs = 100
classifier_learning_rate = 0.001
classifier_categories = ['Artem', 'Stanislav', 'Rodoslav', 'Danil', 'Varvara', 'Polina', 'Yulia', 'Arseny', 'Nastya']
