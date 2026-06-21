import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import json
from PIL import Image
import cv2
import os
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


@st.cache_resource
def load_model_and_labels():
    models_dir = "./Models"
    weights_path = os.path.join(models_dir, "plant_disease_model.weights.h5")
    class_indices_path = os.path.join(models_dir, "class_indices.json")

    with open(class_indices_path, "r") as f:
        class_indices = json.load(f)

    labels = {v: k for k, v in class_indices.items()}
    num_classes = len(labels)

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = False

    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation="softmax")
    ])

    model.build((None, 224, 224, 3))
    model.load_weights(weights_path)

    return model, labels


@st.cache_data
def load_disease_knowledge():
    with open("disease_knowledge_base.json", "r") as f:
        return json.load(f)


def preprocess_image(img):
    img = img.resize((224, 224))
    x = image.img_to_array(img)
    x = x / 255.0
    x = np.expand_dims(x, axis=0)
    return x


def predict(img, model, labels):
    processed_img = preprocess_image(img)
    preds = model.predict(processed_img, verbose=0)

    pred_label_idx = int(np.argmax(preds, axis=1)[0])
    pred_label = labels.get(pred_label_idx, "Unknown")
    confidence = float(preds[0][pred_label_idx])

    return pred_label, confidence


def show_disease_info(label, confidence, disease_info):
    info = disease_info.get(label, {})

    st.success(f"Prediction: {label}")
    st.info(f"Confidence: {confidence * 100:.2f}%")
    st.markdown(f"**Description:** {info.get('description', 'Not available.')}")
    st.markdown(f"**Symptoms:** {info.get('symptoms', 'Not available.')}")
    st.markdown(f"**Management / Solution:** {info.get('management', 'Not available.')}")
    st.markdown(f"**Prevention:** {info.get('prevention', 'Not available.')}")


def evaluate_model_on_dataset(model, val_generator, labels_list):
    preds = model.predict(val_generator, verbose=0)
    y_pred = np.argmax(preds, axis=1)
    y_true = val_generator.classes

    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=labels_list,
        yticklabels=labels_list,
        cmap="Blues",
        ax=ax
    )
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    st.subheader("Classification Report")
    st.text(classification_report(y_true, y_pred, target_names=labels_list))


st.title("Plant Disease Detection")

model, labels = load_model_and_labels()
disease_info = load_disease_knowledge()
labels_list = [labels[i] for i in range(len(labels))]

mode = st.sidebar.radio(
    "Select Mode",
    ["Upload Image", "Live Camera", "Run Dataset Test Cases"]
)


if mode == "Upload Image":
    uploaded_file = st.file_uploader(
        "Upload an image...",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Uploaded Image", use_container_width=True)

        if st.button("Predict Disease"):
            label, conf = predict(img, model, labels)
            show_disease_info(label, conf, disease_info)


elif mode == "Live Camera":
    st.write("Click the checkbox below to start camera prediction.")

    run = st.checkbox("Start Camera")
    frame_window = st.image([])

    if run:
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            st.error("Camera not found or not accessible.")
        else:
            ret, frame = cap.read()

            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(frame_rgb)

                label, conf = predict(img_pil, model, labels)

                text = f"{label}: {conf * 100:.2f}%"
                cv2.putText(
                    frame_rgb,
                    text,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 0, 0),
                    2
                )

                frame_window.image(frame_rgb, channels="RGB")
                show_disease_info(label, conf, disease_info)
            else:
                st.error("Failed to capture image from camera.")

        cap.release()


elif mode == "Run Dataset Test Cases":
    dataset_dir = "./Processed"

    datagen = ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2)

    val_gen = datagen.flow_from_directory(
        dataset_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode="categorical",
        subset="validation",
        shuffle=False
    )

    if st.button("Run Predictions on Dataset and Show Metrics"):
        evaluate_model_on_dataset(model, val_gen, labels_list)

        st.subheader("Sample Predictions")

        for i in range(min(10, val_gen.samples)):
            img_path = val_gen.filepaths[i]
            img = Image.open(img_path).convert("RGB")

            label, conf = predict(img, model, labels)

            st.write(f"Image: {os.path.basename(img_path)}")
            show_disease_info(label, conf, disease_info)
            st.write("---")
