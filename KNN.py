import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score
from matplotlib.colors import ListedColormap
import seaborn as sns

# Veri setini yükleyin
dataset_path = '/kaggle/input/red-wine-quality-cortez-et-al-2009/winequality-red.csv'
dataset = pd.read_csv(dataset_path)

# Bağımsız ve bağımlı değişkenleri ayırın
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values

# Hedef değişkeni kategorilere ayırmak için KBinsDiscretizer kullanın
kbd = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')
y_binned = kbd.fit_transform(y.reshape(-1, 1)).astype(int).ravel()  # type: ignore

# Veriyi eğitim ve test seti olarak ayırın
X_train, X_test, y_train, y_test = train_test_split(X, y_binned, test_size=0.25, random_state=0)

# Özellikleri ölçeklendirin
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# K En Yakın Komşu modelini oluşturun ve eğitin
classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
classifier.fit(X_train, y_train)

# Test seti ile tahmin yapın
y_pred = classifier.predict(X_test)

# Değerlendirme metrikleri
cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# ROC eğrisi ve AUC hesapla (her sınıf için ayrı ayrı)
fpr = {}
tpr = {}
roc_auc = {}

for i in range(len(np.unique(y_binned))):
    fpr[i], tpr[i], _ = roc_curve(y_test, y_pred, pos_label=i)
    roc_auc[i] = auc(fpr[i], tpr[i])

# Değerlendirme metriklerini yazdır
print(f"Confusion Matrix:\n{cm}")
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")

# Confusion matrix'i görselleştir
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Tahmin Edilen')
plt.ylabel('Gerçek (Doğru)')
plt.title('Confusion Matrix')
plt.show()

# ROC eğrisini çiz
plt.figure()
colors = ['blue', 'red', 'green', 'orange', 'purple']
for i in range(len(np.unique(y_binned))):
    plt.plot(fpr[i], tpr[i], color=colors[i], lw=2, label=f'ROC eğrisi sınıfı {i} (area = {roc_auc[i]:.2f})')
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Yanlış Pozitif Oranı')
plt.ylabel('Doğru Pozitif Oranı')
plt.title('Alıcı Çalışma Karakteristiği (ROC) Eğrisi')
plt.legend(loc='center right')
plt.show()

# Eğitim seti için sonuçları görselleştirin
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
                     np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha=0.75, cmap=ListedColormap(('blue', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c=ListedColormap(('yellow', 'green'))(i), label=j)
plt.title('K En Yakın Komşu (Eğitim Seti)')
plt.xlabel('Özellik 1')
plt.ylabel('Özellik 2')
plt.legend()
plt.colorbar()
plt.show()

# Test seti için sonuçları görselleştirin
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
                     np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha=0.75, cmap=ListedColormap(('blue', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
plt.scatter(X_set[:, 0], X_set[:, 1], c=y_set, cmap=ListedColormap(('yellow', 'green')))
plt.title('K En Yakın Komşu (Test Seti)')
plt.xlabel('Özellik 1')
plt.ylabel('Özellik 2')
plt.colorbar()
plt.show()
