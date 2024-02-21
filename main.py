from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score

y_true = [1, 0, 1, 1, 0, 1]
y_pred = [0, 0, 1, 1, 0, 1]

# Korrektklassifikationsrate (Accuracy)
accuracy = accuracy_score(y_true, y_pred)

# Konfusionsmatrix
conf_matrix = confusion_matrix(y_true, y_pred)

# Präzision (Precision)
precision = precision_score(y_true, y_pred)

# Sensitivität (Recall)
recall = recall_score(y_true, y_pred)

# F1 Score
f1 = f1_score(y_true, y_pred)

print(f'Accuracy: {accuracy}')
print(f'Confusion Matrix:\n{conf_matrix}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')
