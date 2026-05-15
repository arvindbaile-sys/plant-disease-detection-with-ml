from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import json
import os
from datetime import datetime

# =======================================================
# 📌 HISTORY SETTINGS & FUNCTIONS (Fixed/Added)
# =======================================================
HISTORY_PATH = 'history.json' # history.json चा path सेट केला आहे

def load_history():
    try:
        with open(HISTORY_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if "users" not in data:
                data["users"] = {}
            return data
    except FileNotFoundError:
        # File नसेल तर सुरुवातीची रचना तयार करा
        return {"users": {}} 
    except json.JSONDecodeError:
        # File corrupt झाल्यास किंवा ती रिकामी असल्यास
        return {"users": {}} 

def save_history(data):
    """Prediction history डेटा JSON फाइलमध्ये जतन करते."""
    # HISTORY_PATH मध्ये write करण्यासाठी
    try:
        with open(HISTORY_PATH, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
    except Exception as e:
        print(f"Error saving history: {e}")

def add_to_history(username, disease, image):
    """युजरच्या history मध्ये नवीन prediction ऍड करते आणि सेव्ह करते."""
    history_data = load_history()
    
    # नवीन prediction item
    new_item = {
        "disease": disease,
        "image": image, # image नाव / path 
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    # युजरची history अपडेट करा (history मध्ये नवीन item सर्वात वर असावे)
    if username not in history_data["users"]:
        history_data["users"][username] = []
    
    # नवीन prediction सर्वात आधी ऍड करा
    history_data["users"][username].insert(0, new_item)
    
    # history file मध्ये डेटा जतन करा
    save_history(history_data)

# =======================================================
# 📦 बाकीचा कोड जसा आहे तसा 
# =======================================================

app = Flask(__name__)

# 🌱 Model आणि class indices path
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'plant_disease_mobilenetv2.h5')
CLASS_INDICES_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'class_indices.json')

# 📦 Model आणि class indices लोड करणे
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(CLASS_INDICES_PATH) as f:
        class_indices = json.load(f)
    classes = list(class_indices.keys())
except Exception as e:
    print(f"Error loading model or class indices: {e}")
    # जर model load झाला नाही तर classes list रिकामी ठेवा
    model = None
    classes = []


# 🌿 Prediction Function
def predict_disease(img_path):
    if not model:
        return "Model_Loading_Error"

    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0
    preds = model.predict(x)
    predicted_class = classes[np.argmax(preds)]
    return predicted_class

# 💚 १५ रोगांची मराठी माहिती
disease_info = {
    "Pepper__bell___Bacterial_spot": {
        "🌿 रोगाचं ओळख आणि कारण": "पानांवर लहान जलद पाणचट ठिपके जे नंतर तपकिरी होतात.",
        "🦠 रोगाचा प्रकार": "Bacterial",
        "📊 रोगाचं प्रमाण": "मध्यम",
        "⚠️ परिणाम": "पानगळ, झाडाची वाढ कमी होते.",
        "💧 पाणी": "पानांवर पाणी टाळा.",
        "💊 उपाय": "कॉपर ऑक्सीक्लोराइड किंवा स्ट्रेप्टोमायसिन फवारणी.",
        "🌾 अतिरिक्त माहिती": "स्वच्छ बियाणे वापरा आणि संक्रमित पाने काढून टाका."
    },

    "Pepper__bell___healthy": {
        "🌿 रोगाचं ओळख आणि कारण": "पानं पूर्णपणे हिरवी आणि निरोगी.",
        "🦠 रोगाचा प्रकार": "Healthy",
        "📊 रोगाचं प्रमाण": "0%",
        "⚠️ परिणाम": "काहीही धोका नाही.",
        "💧 पाणी": "नियमित पण जास्तीचं नाही.",
        "💊 उपाय": "उपायाची गरज नाही.",
        "🌾 अतिरिक्त माहिती": "निरोगी वाढ कायम ठेवण्यासाठी नियमित निरीक्षण."
    },

    "Potato___Early_blight": {
        "🌿 रोगाचं ओळख आणि कारण": "पानांवर तपकिरी गोलाकार डाग ज्यात concentric वर्तुळे दिसतात.",
        "🦠 रोगाचा प्रकार": "Fungal",
        "📊 रोगाचं प्रमाण": "मध्यम ते जास्त",
        "⚠️ परिणाम": "पानगळ, कंदांची वाढ कमी होते.",
        "💧 पाणी": "ओलावा कमी ठेवा.",
        "💊 उपाय": "मॅन्कोझेब किंवा क्लोरोथॅलोनील फवारणी.",
        "🌾 अतिरिक्त माहिती": "जुनी पाने आधी संक्रमित होतात."
    },

    "Potato___Late_blight": {
        "🌿 रोगाचं ओळख आणि कारण": "मोठे पाणचट डाग जे नंतर काळे होतात.",
        "🦠 रोगाचा प्रकार": "Fungal",
        "📊 रोगाचं प्रमाण": "अतिशय जास्त",
        "⚠️ परिणाम": "झाड पूर्ण मरू शकतं, पीक हानी.",
        "💧 पाणी": "पानं ओले राहणार नाहीत याची काळजी.",
        "💊 उपाय": "रिडोमिल गोल्ड किंवा मेटालेक्सील.",
        "🌾 अतिरिक्त माहिती": "थंड आणि ओलसर हवेत जलद पसरतो."
    },

    "Potato___healthy": {
        "🌿 रोगाचं ओळख आणि कारण": "पानं स्वच्छ आणि निरोगी.",
        "🦠 रोगाचा प्रकार": "Healthy",
        "📊 रोगाचं प्रमाण": "0%",
        "⚠️ परिणाम": "काहीही धोका नाही.",
        "💧 पाणी": "नियमित सिंचन.",
        "💊 उपाय": "उपायाची गरज नाही.",
        "🌾 अतिरिक्त माहिती": "कीडरोग नियंत्रण ठेवण्यासाठी निरीक्षण."
    },

    "Tomato_Bacterial_spot": {
        "🌿 रोगाचं ओळख आणि कारण": "लहान, पाणचट डाग जे नंतर गडद तपकिरी होतात.",
        "🦠 रोगाचा प्रकार": "Bacterial",
        "📊 रोगाचं प्रमाण": "मध्यम",
        "⚠️ परिणाम": "फळांवर डाग, उत्पादनात घट.",
        "💧 पाणी": "पानांवर पाणी टाळा.",
        "💊 उपाय": "स्ट्रेप्टोमायसिन किंवा कॉपर फवारणी.",
        "🌾 अतिरिक्त माहिती": "बियाणे निर्जंतुकीकरण आवश्यक."
    },

    "Tomato_Early_blight": {
        "🌿 रोगाचं ओळख आणि कारण": "पानांवर concentric rings असलेले तपकिरी डाग.",
        "🦠 रोगाचा प्रकार": "Fungal",
        "📊 रोगाचं प्रमाण": "जास्त",
        "⚠️ परिणाम": "झाडाची वाढ मंदावते.",
        "💧 पाणी": "ओलावा टाळा.",
        "💊 उपाय": "मॅन्कोझेब फवारणी.",
        "🌾 अतिरिक्त माहिती": "जुनी पाने आधी प्रभावित."
    },

    "Tomato_Late_blight": {
        "🌿 रोगाचं ओळख आणि कारण": "मोठे पाणचट डाग जे काळे पडतात.",
        "🦠 रोगाचा प्रकार": "Fungal",
        "📊 रोगाचं प्रमाण": "अतिशय जास्त",
        "⚠️ परिणाम": "संपूर्ण पीक नष्ट होऊ शकतं.",
        "💧 पाणी": "पाण्याचा थेट संपर्क कमी करा.",
        "💊 उपाय": "रिडोमिल फवारणी.",
        "🌾 अतिरिक्त माहिती": "थंड हवेत वेगाने पसरतो."
    },

    "Tomato_Leaf_Mold": {
        "🌿 रोगाचं ओळख आणि कारण": "पानांच्या खालच्या बाजूस पिवळे ते तपकिरी बुरशीसारखे थर.",
        "🦠 रोगाचा प्रकार": "Fungal",
        "📊 रोगाचं प्रमाण": "मध्यम",
        "⚠️ परिणाम": "पाने कोरडी पडून गळतात.",
        "💧 पाणी": "वायुवीजन वाढवा.",
        "💊 उपाय": "क्लोरोथॅलोनील किंवा कॉपर फवारणी.",
        "🌾 अतिरिक्त माहिती": "घट्ट लागवड टाळा."
    },

    "Tomato_Septoria_leaf_spot": {
        "🌿 रोगाचं ओळख आणि कारण": "पानांवर लहान राखाडी डाग ज्याभोवती गडद किनार.",
        "🦠 रोगाचा प्रकार": "Fungal",
        "📊 रोगाचं प्रमाण": "मध्यम ते जास्त",
        "⚠️ परिणाम": "पाने गळून उत्पादन कमी होते.",
        "💧 पाणी": "पान ओले होणार नाहीत याची दक्षता.",
        "💊 उपाय": "मॅन्कोझेब फवारणी.",
        "🌾 अतिरिक्त माहिती": "संक्रमित पाने काढून जाळा."
    },

    "Tomato_Spider_mites_Two_spotted_spider_mite": {
        "🌿 रोगाचं ओळख आणि कारण": "पानांवर पिवळे ठिपके, जाळ्यासारखा थर.",
        "🦠 रोगाचा प्रकार": "कीड (Mite)",
        "📊 रोगाचं प्रमाण": "जास्त",
        "⚠️ परिणाम": "पानं वाळतात, वाढ थांबते.",
        "💧 पाणी": "पानांवर हलकं पाण्याचं फवारण.",
        "💊 उपाय": "अॅकरा किंवा प्रोपरगाइट फवारणी.",
        "🌾 अतिरिक्त माहिती": "कोरड्या हवेत जास्त वाढतात."
    },

    "Tomato__Target_Spot": {
        "🌿 रोगाचं ओळख आणि कारण": "गोलाकार टार्गेटसारखे डाग.",
        "🦠 रोगाचा प्रकार": "Fungal",
        "📊 रोगाचं प्रमाण": "मध्यम",
        "⚠️ परिणाम": "पाने गळतात.",
        "💧 पाणी": "अतिरिक्त ओलावा टाळा.",
        "💊 उपाय": "क्लोरोथॅलोनील फवारणी.",
        "🌾 अतिरिक्त माहिती": "थोडा ओलावा आणि उष्ण हवेत वाढतो."
    },

    "Tomato__Tomato_YellowLeaf__Curl_Virus": {
        "🌿 रोगाचं ओळख आणि कारण": "पानं पिवळी पडतात आणि आत वळतात.",
        "🦠 रोगाचा प्रकार": "Viral",
        "📊 रोगाचं प्रमाण": "जास्त",
        "⚠️ परिणाम": "फळांची संख्या कमी.",
        "💧 पाणी": "नियमित पण जास्त नाही.",
        "💊 उपाय": "इमिडाक्लोप्रिड फवारणी.",
        "🌾 अतिरिक्त माहिती": "व्हायरसचा इलाज नाही."
    },

    "Tomato__Tomato_mosaic_virus": {
        "🌿 रोगाचं ओळख आणि कारण": "पानांवर मोझॅकसारखे पिवळे-हिरवे ठिपके.",
        "🦠 रोगाचा प्रकार": "Viral",
        "📊 रोगाचं प्रमाण": "मध्यम",
        "⚠️ परिणाम": "वाढ कमी होते.",
        "💧 पाणी": "नियमित.",
        "💊 उपाय": "संक्रमित झाडे हटवा.",
        "🌾 अतिरिक्त माहिती": "हात, साधनांद्वारे पसरतो."
    },

    "Tomato_healthy": {
        "🌿 रोगाचं ओळख आणि कारण": "पानं हिरवी आणि स्वच्छ.",
        "🦠 रोगाचा प्रकार": "Healthy",
        "📊 रोगाचं प्रमाण": "0%",
        "⚠️ परिणाम": "काहीही समस्या नाही.",
        "💧 पाणी": "नियमित पाणी द्या.",
        "💊 उपाय": "गरज नाही.",
        "🌾 अतिरिक्त माहिती": "उत्तम वाढ."
    }
}

# -----------------------------------------------------
# PREDICT ROUTE — FIXED VERSION
# -----------------------------------------------------

@app.route('/predict', methods=['POST'])
def predict_route():
    if 'file' not in request.files:
        return render_template("index.html", error="कृपया image upload करा")

    file = request.files['file']

    if file.filename == '':
        return render_template("index.html", error="कृपया योग्य image निवडा")

    # image save करा
    filepath = os.path.join('static/uploads', file.filename)
    file.save(filepath)

    # prediction मिळवा
    result = predict_disease(filepath)

    # रोगाची माहिती मिळवा
    info = disease_info.get(result, None)
    
    # solution व्हेरिएबलमध्ये convert करा (टेम्पलेटसाठी)
    solution = None
    if info:
        solution = {
            "cause": info.get("🌿 रोगाचं ओळख आणि कारण", ""),
            "type": info.get("🦠 रोगाचा प्रकार", ""),
            "severity": info.get("📊 रोगाचं प्रमाण", ""),
            "effect": info.get("⚠️ परिणाम", ""),
            "water": info.get("💧 पाणी", ""),
            "treatment": info.get("💊 उपाय", ""),
            "extra": info.get("🌾 अतिरिक्त माहिती", "")
        }

    return render_template("index.html", 
                        prediction=result, 
                        solution=solution,  # solution वापरा
                        img=file.filename)


# -----------------------------------------------------
# NEW ROUTES: HISTORY GET AND SAVE (JSON API)
# -----------------------------------------------------

# 💾 प्रेडिक्शन हिस्ट्री जतन करण्यासाठी API
@app.route('/api/save_history', methods=['POST'])
def save_history_route():
    data = request.get_json()
    username = data.get('email')
    disease = data.get('disease')
    image = data.get('image')

    if username and disease and image:
        # add_to_history function कॉल करा
        add_to_history(username, disease, image) 
        return jsonify({"status": "success", "message": "History saved successfully"}), 200

    return jsonify({"status": "error", "message": "Invalid data"}), 400 

# ✅ प्रेडिक्शन हिस्ट्री मिळवण्यासाठी API
@app.route('/api/get_history', methods=['POST'])
def get_history_route():
    data = request.get_json()
    username = data.get('email')

    if username:
        history_data = load_history()
        user_history = history_data["users"].get(username, [])
        return jsonify({"status": "success", "history": user_history}), 200

    return jsonify({"status": "error", "message": "Missing email"}), 400

# ❌ प्रेडिक्शन हिस्ट्री डिलीट करण्यासाठी API
@app.route('/api/clear_history', methods=['POST'])
def clear_history_route():
    data = request.get_json()
    username = data.get('email')
    
    if username:
        history_data = load_history()
        if username in history_data["users"]:
            history_data["users"][username] = []
            save_history(history_data)
            return jsonify({"status": "success", "message": "History cleared successfully"}), 200
        
        return jsonify({"status": "success", "message": "No history found or cleared"}), 200
        
    return jsonify({"status": "error", "message": "Missing email"}), 400

# HOME ROUTE
@app.route('/')
def home():
    return render_template('home.html')

# 🌿 हे इतर routes तुझ्या project नुसार ठेवा
@app.route('/predict', methods=['GET', 'POST'])
def index():
    # इथे prediction logic ठेव
    return render_template('index.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/login')
def login():
    return render_template('login.html')


# MAIN
if __name__ == '__main__':
    app.run(debug=True)