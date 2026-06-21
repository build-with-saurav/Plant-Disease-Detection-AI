import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import json
import os

# Set paths
dataset_dir = '/home/saurav/Desktop/ImageProcessing/Processed'  # Preprocessed images folder
models_dir = '/home/saurav/Desktop/ImageProcessing/Models'
model_save_path = os.path.join(models_dir, 'plant_disease_model.weights.h5')
class_indices_path = os.path.join(models_dir, 'class_indices.json')

# Create models directory if it doesn't exist
os.makedirs(models_dir, exist_ok=True)

# Image parameters
img_height, img_width = 224, 224
batch_size = 32
epochs = 10

# Data preparation using ImageDataGenerator with validation split
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = datagen.flow_from_directory(
    dataset_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

validation_generator = datagen.flow_from_directory(
    dataset_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# Save class indices for inference and documentation
with open(class_indices_path, 'w') as f:
    json.dump(train_generator.class_indices, f)
print(f'Saved class indices to {class_indices_path}')

# Define the model using transfer learning (MobileNetV2 base)
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(img_height, img_width, 3),
    include_top=False,
    weights='imagenet'
)

# Freeze the base model layers during initial training
base_model.trainable = False

# Build the classifier on top of base model
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(train_generator.num_classes, activation='softmax')
])

# Compile model with optimizer, loss, and metrics
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Print model summary for documentation
model.summary()

# Train the model with training and validation data
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=epochs
)

# Save the trained model in HDF5 format for portability
model.save_weights(os.path.join(models_dir, "plant_disease_model.weights.h5"))
print("Model weights saved successfully")
print(f'Model saved to {model_save_path}')

# Optionally, you can add code to save training history, plot metrics, or export to other formats here


