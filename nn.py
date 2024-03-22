import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from genetic_algorithm import GeneticAlgorithm
from sklearn.preprocessing import StandardScaler
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Step 1: Load and preprocess your data
X = np.random.rand(1000, 10)  # Example input features (replace with your actual data)
y = np.random.randint(0, 2, size=(1000,))  # Example target labels (replace with your actual data)

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Step 2: Define your neural network architecture
model = Sequential()
model.add(Dense(units=64, activation='relu', input_dim=10))
model.add(Dense(units=10, activation='softmax'))


# Step 3: Compile your model
model.compile(loss='sparse_categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

# Step 4: Train your model
history = model.fit(X_train_scaled, y_train, epochs=10, batch_size=32, validation_data=(X_val_scaled, y_val))

# Step 5: Evaluate your model
val_loss, val_accuracy = model.evaluate(X_val_scaled, y_val)
print("Validation Loss:", val_loss)
print("Validation Accuracy:", val_accuracy)

# Step 6: Test your model
test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)

# Step 7: Save the trained model with the .keras extension
model.save('trained_model.keras')

# Step 4: Train, validate, and test your model

# Step 4.1: Train your model
history = model.fit(X_train_scaled, y_train, epochs=10, batch_size=32, validation_data=(X_val_scaled, y_val))

# Step 4.2: Evaluate your model on the validation set
val_loss, val_accuracy = model.evaluate(X_val_scaled, y_val)
print("Validation Loss:", val_loss)
print("Validation Accuracy:", val_accuracy)

# Step 4.3: Evaluate your model on the test set
test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)

## Step 5: Evaluate the performance of your model

# Step 5.1: Make predictions on the test set
y_pred = model.predict(X_test_scaled)
y_pred_classes = np.argmax(y_pred, axis=1)

# Step 5.2: Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred_classes)
precision = precision_score(y_test, y_pred_classes, average='weighted')
recall = recall_score(y_test, y_pred_classes, average='weighted')
f1 = f1_score(y_test, y_pred_classes, average='weighted')

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)

# Step 5.3: Generate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred_classes)
print("Confusion Matrix:")
print(conf_matrix)
# Step 1: Make predictions on the validation set
validation_predictions = model.predict(X_val_scaled)
predicted_labels = np.argmax(validation_predictions, axis=1)

# Step 2: Calculate evaluation metrics
accuracy = accuracy_score(y_val, predicted_labels)
precision = precision_score(y_val, predicted_labels, average='macro')  # Use 'macro' for multi-class problems
recall = recall_score(y_val, predicted_labels, average='macro')
f1 = f1_score(y_val, predicted_labels, average='macro')

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)

# Step 3: Generate confusion matrix
conf_matrix = confusion_matrix(y_val, predicted_labels)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()
 # Assuming you have a genetic algorithm implementation

# Step 1: Define the hyperparameter space
hyperparameter_space = {
    'lr': [0.0001, 0.001, 0.01],
    'batch_size': [16, 32, 64],
    'num_layers': [1, 2, 3],
    'neurons_per_layer': [32, 64, 128],
    'activation_functions': ['relu', 'sigmoid', 'tanh'],
    'optimizer': ['sgd', 'adam', 'rmsprop'],
    'dropout_rate': [0.0, 0.1, 0.2],
    'regularization_strength': [0.001, 0.01, 0.1]
}

# Step 2: Define the evaluation function
def evaluate_model(hyperparameters, validation_accuracy=None):
    # Train the model with the given hyperparameters and return the performance metric
    # This function should train the model and return the evaluation metric (e.g., validation accuracy)
    return validation_accuracy

# Step 3: Implement the genetic algorithm
genetic_algorithm = GeneticAlgorithm(generations=20, mutation_rate=0.1)

# Step 4: Run the optimization
best_hyperparameters = genetic_algorithm.optimize(hyperparameter_space, evaluate_model)


# Step 5: Retrieve the best hyperparameters
print("Best Hyperparameters:", best_hyperparameters)

# Step 6: Train the model with the best hyperparameters
# Now, use the best hyperparameters to train the model on the full training data
# Evaluate the model on the test set to assess its performance
# Load the trained model
model = load_model('trained_model.keras')

def preprocess_image(image_path):
    # Read the input image
    image = cv2.imread(image_path)
    # Preprocess the image (resize, convert to grayscale, normalize, etc.)
    # Example:
    resized_image = cv2.resize(image, (224, 224))
    normalized_image = resized_image / 255.0  # Normalize pixel values
    return normalized_image

def predict_image(image_path):
    # Preprocess the input image
    preprocessed_image = preprocess_image(image_path)
    # Make predictions using the model
    predictions = model.predict(np.expand_dims(preprocessed_image, axis=0))
    # Post-process predictions (select class with highest probability)
    predicted_class = np.argmax(predictions)
    probability = predictions[0][predicted_class]
    return predicted_class, probability

def test_on_image(image_path):
    # Make predictions on the input image
    predicted_class, probability = predict_image(image_path)
    # Display results
    print("Predicted Class:", predicted_class)
    print("Probability:", probability)
    # Show the input image
    image = cv2.imread(image_path)
    cv2.imshow("Input Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Test the system on an arbitrary image
test_image_path = 'Facial Recognition Dataset/Testing/Surprise/Suprise-85.jpg'
test_on_image(test_image_path)