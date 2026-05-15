# Evaluation script placeholder
import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'plant_disease_mobilenetv2.h5')
CLASS_INDICES_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'class_indices.json')
TEST_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'dataset', 'test')
IMG_SIZE = (224,224)
BATCH_SIZE = 32


model = tf.keras.models.load_model(MODEL_PATH)
with open(CLASS_INDICES_PATH,'r') as f:
class_indices = json.load(f)


test_datagen = ImageDataGenerator(rescale=1./255)


test_gen = test_datagen.flow_from_directory(TEST_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical', shuffle=False)


preds = model.predict(test_gen)
y_pred = np.argmax(preds, axis=1)
y_true = test_gen.classes
class_names = list(class_indices.keys())


print(classification_report(y_true, y_pred, target_names=class_names))


cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(12,10))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()