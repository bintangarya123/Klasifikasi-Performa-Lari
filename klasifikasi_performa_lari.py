import pandas as pd

# Membaca dataset Excel
df = pd.read_excel("Dataset Aktivitas Lari.xlsx")

# Menampilkan 5 data pertama
print(df.head())

# Menampilkan Informasi Struktur Data
print(df.info())

# Mengubah waktu menjadi detik
df["Waktu_Detik"] = df["Waktu"].apply(
    lambda x: int(str(x).split(":")[0]) * 3600 +
              int(str(x).split(":")[1]) * 60 +
              int(str(x).split(":")[2])
)

print(df[["Waktu", "Waktu_Detik"]].head())

# Mengubah pace menjadi detik
df["Pace_Detik"] = df["Pace (Menit/Km)"].apply(
    lambda x: int(str(x).split(":")[0]) * 60 +
              int(str(x).split(":")[1])
)

print(df[["Pace (Menit/Km)", "Pace_Detik"]].head())

# Memeriksa missing values
print(df.isnull().sum())

# Memeriksa data duplikat
print(df.duplicated().sum())

# Menyimpan dataset hasil persiapan
df.to_csv("Dataset Aktivitas Lari Final 50.csv", index=False)

# Menentukan skor jarak
df["Skor_Jarak"] = df["Jarak (Km)"].apply(
    lambda x: 1 if x <= 7 else
              2 if x <= 14 else
              3
)

# Menentukan skor waktu
df["Skor_Waktu"] = df["Waktu_Detik"].apply(
    lambda x: 1 if x <= 2700 else
              2 if x <= 5400 else
              3
)

# Menentukan skor pace
df["Skor_Pace"] = df["Pace_Detik"].apply(
    lambda x: 1 if x > 480 else
              2 if 360 <= x <= 480 else
              3
)

# Menentukan skor detak jantung
df["Skor_HR"] = df["Rata-rata Detak Jantung"].apply(
    lambda x: 1 if x < 140 else
              2 if x <= 160 else
              3
)

df["Total_Skor"] = (
    df["Skor_Jarak"] +
    df["Skor_Waktu"] +
    df["Skor_Pace"] +
    df["Skor_HR"]
)

# Menentukan label klasifikasi
df["Label"] = df["Total_Skor"].apply(
    lambda x: "Rendah" if x <= 6 else
              "Sedang" if x <= 9 else
              "Tinggi"
)

print(df[[
    "Nama",
    "Skor_Jarak",
    "Skor_Waktu",
    "Skor_Pace",
    "Skor_HR",
    "Total_Skor",
    "Label"
]].head())

# Menentukan variabel independen (fitur)
X = df[[
    "Jarak (Km)",
    "Waktu_Detik",
    "Pace_Detik",
    "Rata-rata Detak Jantung"
]]

print(X.head())

# Menentukan variabel dependen (label)
y = df["Label"]

print(y.head())

from sklearn.model_selection import train_test_split

# Membagi data training dan testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Jumlah Data Training :", len(X_train))
print("Jumlah Data Testing :", len(X_test))

print(y_train.value_counts())
print()

print(y_test.value_counts())


from sklearn.naive_bayes import GaussianNB

print("Library GaussianNB berhasil diimpor.")


model = GaussianNB()

print(model)


model.fit(X_train, y_train)

print("Model berhasil dilatih.")


y_pred = model.predict(X_test)

print(y_pred)

hasil_prediksi = pd.DataFrame({
    "Aktual": y_test.values,
    "Prediksi": y_pred
})

print(hasil_prediksi)



from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Membuat confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Menampilkan nilai confusion matrix
print(cm)

# Visualisasi confusion matrix
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Rendah", "Sedang", "Tinggi"]
)

disp.plot(cmap="Blues")

plt.show()


from sklearn.metrics import accuracy_score

# Menghitung akurasi model
akurasi = accuracy_score(y_test, y_pred)

# Menampilkan nilai akurasi
print("Accuracy :", akurasi)


from sklearn.metrics import classification_report

# Menampilkan hasil evaluasi model
print(classification_report(y_test, y_pred))

import joblib

# Menyimpan model Gaussian Naive Bayes
joblib.dump(model, "model_gnb.pkl")


