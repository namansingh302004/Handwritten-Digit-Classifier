import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from scipy.ndimage import rotate, shift, zoom

# Load and normalize MNIST dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

# Parameters: specify the ranges for transformations
shift_range_x = (-4, 4) 
shift_range_y = (-4, 4) 
rotation_range = (-20, 20)
scale_range = (0.8, 1.2) 
noise_prob = (0.01, 0.03)

def scale_image(image, scale_factor):
    zoomed = zoom(image, scale_factor)
    
    h_zoom, w_zoom = zoomed.shape
    h, w = image.shape
    
    if h_zoom > h:
        start_h = (h_zoom - h) // 2
        start_w = (w_zoom - w) // 2
        zoomed = zoomed[start_h:start_h + h, start_w:start_w + w]
    else:
        pad_h = (h - h_zoom) // 2
        pad_w = (w - w_zoom) // 2
        zoomed = np.pad(zoomed, ((pad_h, h - h_zoom - pad_h), (pad_w, w - w_zoom - pad_w)), mode='constant', constant_values=0)
    
    return zoomed

def add_salt_and_pepper_noise(image, prob):
    noisy_image = np.copy(image)
    num_salt = np.ceil(prob * image.size * 0.5)
    num_pepper = np.ceil(prob * image.size * 0.5)

    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    noisy_image[coords[0], coords[1]] = np.random.uniform(*(0, 1))

    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    noisy_image[coords[0], coords[1]] = np.random.uniform(*(0, 1))

    return noisy_image

def translate_image(image, shift_x, shift_y):
    return shift(image, [shift_y, shift_x], mode='constant', cval=0)

def rotate_image(image, angle):
    return rotate(image, angle, reshape=False, mode='constant', cval=0)

def augment_image(image):
    # Random translation
    shift_x = np.random.uniform(*shift_range_x)
    shift_y = np.random.uniform(*shift_range_y)
    translated = translate_image(image, shift_x=shift_x, shift_y=shift_y)
    
    # Random rotation
    angle = np.random.uniform(*rotation_range)
    rotated = rotate_image(translated, angle=angle)
    
    # Random scaling
    scale_factor = np.random.uniform(*scale_range)
    scaled = scale_image(rotated, scale_factor)
    
    # Add random noise
    prob = np.random.uniform(*noise_prob)
    noisy_image = add_salt_and_pepper_noise(scaled, prob=prob)
    
    return noisy_image

'''
# Display a 5x5 grid of test images
fig, axes = plt.subplots(10, 10, figsize=(10, 10))
axes = axes.ravel()

# Apply augmentations and show the results
for i in range(100):
    x_test[i] = augment_image(x_test[i])
    axes[i].imshow(x_test[i], cmap='gray')  # Show each image in grayscale
    axes[i].set_title(f"Label: {y_test[i]}")  # Show the label as the title
    axes[i].axis('off')  # Hide the axes

plt.tight_layout()
plt.show()
'''