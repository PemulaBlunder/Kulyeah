# Regresi pada data CO2

1. Download DataCO2.csv, lakukan eksplorasi data dan preprocessing yang menurut anda 
diperlukan.
2. Apakah ada tipe data non numerik pada DataCO2.csv? Dan apakah ada yang bersifat 
nominal/ordinal? Jika ada sebutkan kolom mana saja, jika tidak jelaskan mengapa.  
3. Buatlah model regresi menggunakan metode yang paling tepat menurut anda.  
4. Screen shoot kode python yang anda gunakan untuk menjawab pertanyaan 6-8

## 1. Lakukan eksplorasi data dan preprocessing yang menurut anda diperlukan.
```py
df1.info()
```
![image](https://github.com/user-attachments/assets/a3d22d10-eadc-41b2-b51f-34939fa5c29d)
Melihat dari gambar tidak ada nilai NaN dalam data CO2

```py
numerik=[]
for i in df1.columns.to_list():
    if df1[i].dtypes!='O':
        numerik.append(i)
```
Syntax tersebut akan menghasilkan kolom mana yang memiliki tipe data numerik

## 2. Apakah ada tipe data non numerik pada DataCO2.csv? Dan apakah ada yang bersifat nominal/ordinal? Jika ada sebutkan kolom mana saja, jika tidak jelaskan mengapa.
Ya, ada tipe data non numerik. Dan kedua kolom yang memiliki data non numerik, bersifat nominal yaitu kolom `Model & Nama`

## 3. Buatlah model regresi menggunakan metode yang paling tepat menurut anda.
```py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

X = np.array([df1['Bobot'], df1['Volume']]).T
y = np.array(df1['CO2'])

# Membagi df1set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.235, random_state=42)

# Membuat model regresi linear
model = LinearRegression()
model.fit(X_train, y_train)

# Prediksi pada df1 uji
y_pred_test = model.predict(X_test)

# Evaluasi model
mse = mean_squared_error(y_test, y_pred_test)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred_test)
r2 = r2_score(y_test, y_pred_test)

print("MSE:", mse)
print("RMSE:", rmse)
print("MAE:", mae)
print("R-squared:", r2)

# Visualisasi
plt.scatter(X_test[:, 0], y_test, color='blue', label='df1 Asli')
plt.scatter(X_test[:, 0], y_pred_test, color='red', label='Prediksi')
plt.title('Multiple Linear CO2')
plt.ylabel('CO2 Yang dihasilkan')
plt.legend()
plt.grid(True)
plt.show()
```
![image](https://github.com/user-attachments/assets/fdaf48db-3803-4579-bc59-c694006916b8)

`MSE: 51.698353637313716`

`RMSE: 7.190156718550279`

`MAE: 5.824670969484398`

`R-squared: 0.37796098564729486`

Dari hasil R-square, regresi antara kolom (Volume, Bobot) kurang tepat digunakan untuk regresi multivarian
