import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Modeli önceden yükle
model_path = "bone_fracture_mobilenetv2_initial.h5"  # Bu yolu kendi modelinize göre değiştirebilirsiniz
model = tf.keras.models.load_model(model_path)

# Başlık ve açıklamalar
st.title("Kemik Kırığı Tespiti Modeli")
st.write("Bu uygulama, eğitilmiş bir model kullanarak kemik kırığı olup olmadığını tespit eder.")

# Kullanıcıdan görsel yüklemeyi iste
uploaded_file = st.file_uploader("Bir görsel yükleyin", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Görseli yükle ve ön işleme
    img = image.load_img(uploaded_file, target_size=(224, 224))  # Görseli boyutlandır
    img_array = image.img_to_array(img)  # Görseli numpy array'e çevir
    img_array = np.expand_dims(img_array, axis=0)  # Batch boyutunu ekle

    # Görseli normalize et
    img_array = preprocess_input(img_array)  # MobileNetV2 için preprocess

    # Görseli ekranda göster
    st.image(img, caption="Yüklenen Görsel", use_column_width=True)

    # Tahmin yap
    prediction = model.predict(img_array)[0][0]
    class_labels = ['fracture', 'normal']  # 0 -> fracture, 1 -> normal
    predicted_class = class_labels[int(prediction > 0.5)]
    confidence = prediction if prediction > 0.5 else 1 - prediction

    # Sonuçları daha büyük ve ortalanmış şekilde ekrana yazdır
    st.markdown(f"""
    <h2 style="text-align: center; color: #4CAF50;">Tahmin edilen sınıf: {predicted_class}</h2>
    <h3 style="text-align: center; color: #FF6347;">{confidence:.2%} güven</h3>
    """, unsafe_allow_html=True)
