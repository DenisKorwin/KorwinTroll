from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score

y_true = [1, 0, 1, 1, 0, 1]
y_pred = [0, 0, 1, 1, 0, 1]


# Korrektklassifikationsrate (Accuracy)
"""
Misst den Anteil der Gesamtzahl von Vorhersagen, die korrekt waren.
Es wird berechnet durch die Summe der wahren Positiven und wahren Negativen geteilt durch
die Gesamtzahl der Fälle.
"""
accuracy = accuracy_score(y_true, y_pred)


# Konfusionsmatrix
"""
Eine Tabelle, die verwendet wird, um die Leistung eines Klassifikationsmodells zu visualisieren.
Sie zeigt die Anzahl der korrekten und falschen Vorhersagen aufgeteilt nach Klassen.

Die Hauptachsen sind:
True Positives (TP): Korrekt positiv klassifizierte Fälle
True Negatives (TN): Korrekt negativ klassifizierte Fälle
False Positives (FP): Fälschlicherweise als positiv klassifizierte Fälle
False Negatives (FN): Fälschlicherweise als negativ klassifizierte Fälle
"""
conf_matrix = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = conf_matrix.ravel()


# Prävelenz, Relevanz (Precision), Sensitivität (Recall), Spezifität
"""
Prävalenz: Anteil der tatsächlichen Positiven in der Population.
Präzision (Precision): Anteil der wahren Positiven an allen als positiv klassifizierten Fällen (TP / (TP + FP)).
Sensitivität (Recall): Anteil der wahren Positiven an allen tatsächlich positiven Fällen (TP / (TP + FN)).
Spezifität: Anteil der wahren Negativen an allen tatsächlich negativen Fällen (TN / (TN + FP)).
"""
prevalence = sum(y_true) / len(y_true)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
specifity = tn / (tn + fp)


# F1 Score
"""
Ein Maß, das die Balance zwischen Präzision und Recall berücksichtigt.
Er ist das harmonische Mittel von Präzision und Recall und wird berechnet durch:

f1 = 2 * ((precision * recall) / (precision + recall))
"""
f1 = f1_score(y_true, y_pred)



print(f"Accuracy: {accuracy}")
print(f"Confusion matrix: \n{conf_matrix}")
print(f"tn, fp, fn, tp: {tn}, {fp}, {fn}, {tp}")
print(f"Prevalence: {prevalence}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"Specifity: {specifity}")
print(f"F1 Score: {f1}")
