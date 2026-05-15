# Flask API for inference placeholder
from flask import Flask, request, jsonify
import tensorflow as tf
from PIL import Image
import numpy as np
import io, json, os


app = Flask(__name__)
BASE = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE, '..', 'models', 'plant_disease_mobilenetv2.h5')
CLASS_INDICES_PATH = os.path.join(BASE, '..', 'models', 'class_indices.json')
KB_PATH = os.path.join(BASE, 'knowledge_base.json')


model = tf.keras.models.load_model(MODEL_PATH)
with open(CLASS_INDICES_PATH,'r') as f:
class_indices = json.load(f)
inv_map = {v:k for k,v in class_indices.items()}
with open(KB_PATH,'r') as f:
kb = json.load(f)


IMG_SIZE = (224,224)


def preprocess_image(image_bytes):
img = Image.open(io.BytesIO(image_bytes)).convert('RGB').resize(IMG_SIZE)
arr = np.array(img)/255.0
arr = np.expand_dims(arr, axis=0)
return arr


@app.route('/predict', methods=['POST'])
def predict():
if 'file' not in request.files:
return jsonify({'error':'no file provided'}), 400
f = request.files['file']
img_bytes = f.read()
x = preprocess_image(img_bytes)
preds = model.predict(x)[0]
idx = int(np.argmax(preds))
label = inv_map[idx]
confidence = float(np.max(preds))
kb_entry = kb.get(label, {})
return jsonify({
'label': label,
'confidence': confidence,
'symptoms': kb_entry.get('symptoms', []),
'remedies': kb_entry.get('remedies', [])
})


if __name__ == '__main__':
app.run(debug=True, host='0.0.0.0', port=5000)