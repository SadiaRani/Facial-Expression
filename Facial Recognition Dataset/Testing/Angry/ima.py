import cv2
import matplotlib.pyplot as plt

# Load an original image
original_image = cv2.imread(r"C:\Users\lenovo\PycharmProjects\Opencvproject\Facial Recognition Dataset\Testing\Angry\Angry-109.jpg", cv2.IMREAD_GRAYSCALE)

# Load the corresponding normalized image
normalized_image = cv2.imread(r"C:\Users\lenovo\PycharmProjects\Opencvproject\Facial Recognition Dataset\Testing\Angry\Angry-109.jpg", cv2.IMREAD_GRAYSCALE)

# Display original and normalized images side by side
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(original_image, cmap='gray')
plt.title('Original Image')
plt.subplot(1, 2, 2)
plt.imshow(normalized_image, cmap='gray')
plt.title('Normalized Image')
plt.show()
