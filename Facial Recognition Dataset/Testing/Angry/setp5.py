from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

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

# Step 4: Visualize confusion matrix (optional)
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()
