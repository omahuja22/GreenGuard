import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# load trained model
model = tf.keras.models.load_model("../model/plant_disease_model.keras")

class_names = [
"Pepper__bell___Bacterial_spot",
"Pepper__bell___healthy",
"Potato___Early_blight",
"Potato___healthy",
"Potato___Late_blight",
"Tomato_Bacterial_spot",
"Tomato_Early_blight",
"Tomato_healthy",
"Tomato_Late_blight",
"Tomato_Leaf_Mold",
"Tomato_Septoria_leaf_spot",
"Tomato_Spider_mites_Two_spotted_spider_mite",
"Tomato__Target_Spot",
"Tomato__Tomato_mosaic_virus",
"Tomato__Tomato_YellowLeaf__Curl_Virus"
]

st.title("🌱 GreenGuard")
st.write("Plant Disease Detection using Machine Learning")

uploaded_file = st.file_uploader("Upload a plant leaf image")

if uploaded_file is not None:

    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image")

    img = image.resize((224,224))
    img_array = np.array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)

    predicted_class = class_names[np.argmax(prediction)]

    st.success(f"Predicted Disease: {predicted_class}")

    