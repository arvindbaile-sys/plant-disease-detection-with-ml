# 🌿 Plant Disease Detection using MobileNetV2

A deep learning-based web application to detect diseases in **Tomato**, **Potato**, and **Bell Pepper** leaves. The system provides disease name, severity, treatment suggestions, and prevention tips in **Marathi** language.

> 🚀 **Live Demo:** (Coming soon)  
> 📦 **GitHub Repo:** [arvindbaile-sys/plant-disease-detection-with-ml](https://github.com/arvindbaile-sys/plant-disease-detection-with-ml)  
> 🧠 **Dataset:** [Plant Disease Detection Dataset on Kaggle](https://www.kaggle.com/datasets/arvindbaile/plant-disease-detection-dataset)

---

## 📌 Features

- ✅ **15 classes** including healthy leaves and common diseases (Early blight, Late blight, Bacterial spot, Leaf mold, Spider mites, Mosaic virus, etc.)
- ✅ **MobileNetV2** transfer learning model (trained on PlantVillage dataset)
- ✅ **Flask web interface** with responsive design
- ✅ **Marathi language support** for disease information and remedies
- ✅ **User authentication** (Login/Register) with password update
- ✅ **Prediction history** stored per user (JSON file)
- ✅ **Contact form** with EmailJS integration
- ✅ **Background video** and attractive UI

---

## 🧠 Model Performance

| Metric | Value |
|--------|-------|
| Architecture | MobileNetV2 (fine-tuned) |
| Input size | 224x224 |
| Training accuracy | ~96% |
| Validation accuracy | ~94% |
| Number of classes | 15 |

--- 

## 📂 Project Structure
```text 
plant-disease-detection-with-ml/
├── app.py # Main Flask application
├── train.py # Model training script
├── evaluate.py # Model evaluation script
├── inference_api.py # Standalone inference API
├── predict.py # Single image prediction
├── requirements.txt # Python dependencies
├── .gitignore # Ignored files (models, videos, etc.)
├── README.md
├── LICENSE
├── templates/ # HTML templates (home, index, login, contact)
├── static/ # CSS, icons, background images, uploads
├── models/ # Trained model (.h5) and class_indices.json
├── data/ # Dataset (ignored in git)
├── frontend/ # Original frontend files (optional)
└── src/ # Source code backup
```

---

## 🔧 Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/arvindbaile-sys/plant-disease-detection-with-ml.git
cd plant-disease-detection-with-ml

```

## 2.Create a virtual environment (optional but recommended)
```text 
bash
python -m venv venv
source venv/bin/activate      # Linux/Mac
venv\Scripts\activate         # Windows
```

## 3.Install dependencies
```text
pip install -r requirements.txt
```

## 4.Download the trained model
```text
The model file plant_disease_mobilenetv2.h5 is not included in the repository due to size limits.
You can either:

Train the model yourself using train.py (requires the dataset)

Download the pre-trained model from [Google Drive / Release page] (link to be added)

```

## 5.Run the Flask app
```text
python app.py
```
Then open http://localhost:5000 in your browser.

## 6.🌱Usage
```text
Register / Login using email and password (localStorage based).

Upload a leaf image (JPG/PNG) from the Predict page.

The model will identify the disease and display:

Disease name

Type (Fungal/Bacterial/Viral/Healthy)

Severity level

Water management tips

Recommended treatment

Additional information (all in Marathi)

View your prediction history in the user profile dropdown.

Contact the developer via the Contact page (EmailJS).

```

## 7.📊 Dataset

The dataset used for training is available on Kaggle:

Plant Disease Detection Dataset by Arvind Baile

It contains 15 classes of healthy and diseased leaves of Tomato, Potato, and Bell Pepper.
The dataset is a curated subset of the PlantVillage dataset with additional augmentations.


## 8.🛠️ Technologies Used

Backend: Flask, TensorFlow / Keras

Frontend: HTML5, CSS3, JavaScript (vanilla)

Authentication: LocalStorage (frontend-only)

Email Service: EmailJS

Version Control: Git + GitHub

Model: MobileNetV2 (pre-trained on ImageNet)



## 9.📝 License
```text
This project is licensed under the MIT License – see the LICENSE file for details.
```

## 10.🙏 Acknowledgements

PlantVillage Dataset

TensorFlow

Kaggle

Flask



## 11.📧 Contact
```text
Author: Arvind Baile
Email: arvindbaile@gmail.com
Instagram: @me_ek_shetkari07
YouTube: Agventure Urban Farming
```

## 12.⭐ If you like this project, please give it a star on GitHub!
## 13.🐛 For issues or suggestions, feel free to open an issue.


---

## ✅ तू काय करायचं?

1. ही संपूर्ण content कॉपी कर.
2. तुझ्या प्रोजेक्ट फोल्डरमध्ये `README.md` फाइल उघड (नसेल तर नवीन तयार कर).
3. पेस्ट कर आणि सेव्ह कर.
4. नंतर Git वर push कर:

```bash
git add README.md
git commit -m "Add comprehensive README"
git push
