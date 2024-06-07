import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
import os
# Load the model
MODEL_PATH = os.path.join(os.getcwd(),"api\modeloffline")
model = hub.KerasLayer(MODEL_PATH)

diseases = {
    "0": "Apple___Apple_scab",
    "1": "Apple___Black_rot",
    "2": "Apple___Cedar_apple_rust",
    "3": "Apple___healthy",
    "4": "Blueberry___healthy",
    "5": "Cherry_(including_sour)___Powdery_mildew",
    "6": "Cherry_(including_sour)___healthy",
    "7": "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "8": "Corn_(maize)___Common_rust_",
    "9": "Corn_(maize)___Northern_Leaf_Blight",
    "10": "Corn_(maize)___healthy",
    "11": "Grape___Black_rot",
    "12": "Grape___Esca_(Black_Measles)",
    "13": "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "14": "Grape___healthy",
    "15": "Orange___Haunglongbing_(Citrus_greening)",
    "16": "Peach___Bacterial_spot",
    "17": "Peach___healthy",
    "18": "Pepper_bell___Bacterial_spot",
    "19": "Pepper_bell___healthy",
    "20": "Potato___Early_blight",
    "21": "Potato___Late_blight",
    "22": "Potato___healthy",
    "23": "Raspberry___healthy",
    "24": "Soybean___healthy",
    "25": "Squash___Powdery_mildew",
    "26": "Strawberry___Leaf_scorch",
    "27": "Strawberry___healthy",
    "28": "Tomato___Bacterial_spot",
    "29": "Tomato___Early_blight",
    "30": "Tomato___Late_blight",
    "31": "Tomato___Leaf_Mold",
    "32": "Tomato___Septoria_leaf_spot",
    "33": "Tomato___Spider_mites Two-spotted_spider_mite",
    "34": "Tomato___Target_Spot",
    "35": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "36": "Tomato___Tomato_mosaic_virus",
    "37": "Tomato___healthy"
}

print("len : ", len(diseases.keys()))
# Define a function to predict the disease
import tensorflow as tf
from PIL import Image

# Assuming `model` and `diseases` dictionary are already defined

def predict_disease(image_path):
    # Load the image and preprocess it
    img = Image.open(image_path)
    img = img.resize((224, 224))
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    img = tf.expand_dims(img, axis=0)
    
    # Make the prediction
    prediction = model(img)
    prediction_probabilities = tf.nn.softmax(prediction, axis=1)
    predicted_class = tf.argmax(prediction_probabilities, axis=1)
    
    # Extract the predicted class index and confidence
    idx = predicted_class[0].numpy()
    confidence = prediction_probabilities[0][idx].numpy()

    return diseases.get(str(idx), "Unknown"), float(confidence)



