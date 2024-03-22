
import numpy as np
from skimage import io
from sklearn.preprocessing import MinMaxScaler

# Load the images
image_filenames = []  # List of image file paths
images = [io.imread(filename) for filename in image_filenames]

# Convert images to numpy arrays
images_array = np.array(images)

# Flatten the images (if needed)
# images_flat = images_array.reshape(images_array.shape[0], -1)

# Apply Min-Max Scaling
scaler = MinMaxScaler()
normalized_images = scaler.fit_transform(images_array.reshape(images_array.shape[0], -1))

# Reshape the normalized images back to their original shape
normalized_images = normalized_images.reshape(images_array.shape)

# Now, 'normalized_images' contains the normalized dataset of images
