import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import seaborn as sns
from sklearn.preprocessing import label_binarize

# Veri setini yükle
dataset_path = '/kaggle/input/red-wine-quality-cortez-et-al-2009/winequality-red.csv'
veri = pd.read_csv(dataset_path)

# Bağımsız değişkenleri seç
ozellikler = veri.iloc[:, [3, 4]].values
# Gerçek etiketleri al (örneğin kalite değerleri)
gercek_etiketler = veri.iloc[:, -1].values  # Son sütun, hedef değişkenimiz

# Optimal küme sayısı ile KMeans'i veri setine uygula
kmeans_modeli = KMeans(n_clusters=5, init='k-means++', max_iter=300, n_init=10, random_state=0)
tahmin_edilen_etiketler = kmeans_modeli.fit_predict(ozellikler)

# Confusion matrix oluştur
karisiklik_matrisi = confusion_matrix(gercek_etiketler, tahmin_edilen_etiketler)
print("Karişıklık Matrisi:")
print(karisiklik_matrisi)

# Confusion matrix'i görselleştir
plt.figure(figsize=(10, 7))
sns.heatmap(karisiklik_matrisi, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Tahmin Edilen')
plt.ylabel('Gerçek (Doğru)')
plt.title('Karişıklık Matrisi')
plt.show()

# Sınıflandırma raporu
siniflandirma_raporu = classification_report(gercek_etiketler, tahmin_edilen_etiketler)
print("Sınıflandırma Raporu:")
print(siniflandirma_raporu)

# Kümelemeleri görselleştir
plt.scatter(ozellikler[tahmin_edilen_etiketler == 0, 0], ozellikler[tahmin_edilen_etiketler == 0, 1], s=100, c='red', label='Küme 1')
plt.scatter(ozellikler[tahmin_edilen_etiketler == 1, 0], ozellikler[tahmin_edilen_etiketler == 1, 1], s=100, c='blue', label='Küme 2')
plt.scatter(ozellikler[tahmin_edilen_etiketler == 2, 0], ozellikler[tahmin_edilen_etiketler == 2, 1], s=100, c='green', label='Küme 3')
plt.scatter(ozellikler[tahmin_edilen_etiketler == 3, 0], ozellikler[tahmin_edilen_etiketler == 3, 1], s=100, c='cyan', label='Küme 4')
plt.scatter(ozellikler[tahmin_edilen_etiketler == 4, 0], ozellikler[tahmin_edilen_etiketler == 4, 1], s=100, c='magenta', label='Küme 5')
plt.scatter(kmeans_modeli.cluster_centers_[:, 0], kmeans_modeli.cluster_centers_[:, 1], s=300, c='yellow', label='Küme Merkezleri')
plt.title('K-means Kümeleme')
plt.xlabel('Özellik 1')
plt.ylabel('Özellik 2')
plt.legend()
plt.show()

# ROC Eğrisi Hesaplama ve Görselleştirme
gercek_etiketler_binarize = label_binarize(gercek_etiketler, classes=np.unique(gercek_etiketler))
sinif_sayisi = gercek_etiketler_binarize.shape[1]

# K-means tahminlerini binarize etme
tahmin_edilen_etiketler_binarize = label_binarize(tahmin_edilen_etiketler, classes=np.unique(tahmin_edilen_etiketler))

fpr = {}
tpr = {}
roc_auc = {}

for i in range(sinif_sayisi):
    fpr[i], tpr[i], _ = roc_curve(gercek_etiketler_binarize[:, i], tahmin_edilen_etiketler_binarize[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# ROC eğrisi çizimi
plt.figure()
colors = ['aqua', 'darkorange', 'cornflowerblue', 'red', 'green', 'blue', 'yellow', 'black']
for i in range(sinif_sayisi):
    plt.plot(fpr[i], tpr[i], color=colors[i], lw=2, label=f'ROC sınıf {i} (alan = {roc_auc[i]:.2f})')
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Yanlış Pozitif Oranı')
plt.ylabel('Doğru Pozitif Oranı')
plt.title('ROC Eğrisi')
plt.legend(loc='center right')
plt.show()
