import tensorflow as tf

import hyperparams
from classifier import Classifier


dataset, labels = Classifier.get_json_dataset()
model = Classifier.get_model_architecture()

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hyperparams.classifier_learning_rate),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

model.fit(
    x=dataset,
    y=labels,
    batch_size=hyperparams.classifier_batch_size,
    validation_split=hyperparams.classifier_val_split,
    epochs=hyperparams.classifier_epochs
)

model.save_weights(hyperparams.model_face_classification_path)
