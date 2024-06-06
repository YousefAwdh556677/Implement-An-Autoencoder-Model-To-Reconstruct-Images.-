import numpy as np
import cv2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
import matplotlib.pyplot as plt

def load_and_preprocess_images(paths, size=(128, 128)):
    images = []
    for path in paths:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE) / 255.0
        img = cv2.resize(img, size)
        images.append(img)
    return np.stack(images)


def build_autoencoder(input_shape):
    input_layer = Input(shape=input_shape)

    # Encoder
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    encoded_layer = MaxPooling2D((2, 2), padding='same')(x)

    # Decoder
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(encoded_layer)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    output_layer = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    # Create autoencoder model
    autoencoder_model = Model(input_layer, output_layer)
    autoencoder_model.compile(optimizer='adam', loss='binary_crossentropy')
    return autoencoder_model

def plot_comparison(original, reconstructed, epoch):
    plt.figure(figsize=(6, 3))

    # Original image
    ax = plt.subplot(1, 2, 1)
    plt.title(f'Original Image')
    plt.imshow(original.squeeze(), cmap='gray')
    plt.axis('off')

    # Reconstructed image
    ax = plt.subplot(1, 2, 2)
    plt.title(f'Reconstructed Image (Epoch {epoch})')
    plt.imshow(reconstructed.squeeze(), cmap='gray')
    plt.axis('off')

    plt.show()

def train_and_reconstruct(autoencoder_model, x_train, img_to_reconstruct, epochs_to_check):
    for epoch in epochs_to_check:
        # Train the autoencoder
        autoencoder_model.fit(x_train, x_train, epochs=epoch, batch_size=1, shuffle=True, verbose=0)

        # Reconstruct the image
        reconstructed_img = autoencoder_model.predict(np.expand_dims(img_to_reconstruct, axis=0))

        # Plot the comparison
        plot_comparison(img_to_reconstruct, reconstructed_img, epoch)

# Paths to images
image_paths = [
    '/petra1.jpg',
    '/petra2.jpg',
    '/petra3.JPG',
    '/petra4.jpg'
]

# Load and preprocess images
images = load_and_preprocess_images(image_paths)
x_train = np.expand_dims(images[1:], axis=-1)  # Use petra2, petra3, petra4 for training
img1 = np.expand_dims(images[0], axis=-1)  # Use petra1 for reconstruction

# Build the autoencoder
input_shape = (128, 128, 1)
autoencoder_model = build_autoencoder(input_shape)
autoencoder_model.summary()

# Train and reconstruct images
epochs_to_check = [1, 10, 50, 100]
train_and_reconstruct(autoencoder_model, x_train, img1, epochs_to_check)
