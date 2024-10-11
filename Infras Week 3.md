# Eksplorasi Mandiri

1. Summary Statistic dari Data
2. Eksplorasi Data, sajikan data yang mungkin dalam beragam diagram chart
3. Lakukan Uji Korelasi
   
## Import Library dan Memuat Data
```py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr

data=pd.read_csv('C:\\Users\\Iyan\\Downloads\\old_cars.csv')
data
```
- Library Pandas digunakan untuk memuat dataframe
- Library Numpy digunakan untuk fungsi matematis
- Library Matplotlib digunakan untuk visualisasi
- Library Seaborn digunakan untuk visualisasi juga

Untuk menemukan direktori file kita dapat men-select file tersebut dan akan ada simbol ... kemudian kita pilih yang copy path.

![image](https://github.com/user-attachments/assets/c42fb41f-1b09-4ec0-96da-f770362571a7)

Setelah anda memanggil data seharusnya akan mengeluarkan output seperti ini.

![image](https://github.com/user-attachments/assets/ab3cc885-0fc8-4746-af24-f0b76200cd8f)

## 1. Summary Statistic Data
```py
print(f"Data tersebut memeiliki frekuensi sebanyak {len(data)}\n")
kolom=data.columns.to_list()
for y in kolom:
    if data[y].dtype == 'O':
        continue
    print(f'Dari kolom {y} nilai mean (rata-rata)',(data[y].mean()))
    print(f'Dari kolom {y} nilai median nya',(data[y].median()))
    print(f'Dari kolom {y} nilai modus nya',(data[y].mode()[0]))
    print(f'Dari kolom {y} nilai varians nya',(data[y].std(ddof=1)))
    print(f'Dari kolom {y} nilai standar deviasi nya',(data[y].var(ddof=1)),'\n')
```
Setelah anda run kode tersebut output akan mengeluarkan

![image](https://github.com/user-attachments/assets/2982361c-0d05-40af-ba07-09e42dc660dc)

Sebelum melakukan looping kita pisahkan terlebih dahulu kolom yang tipenya `'Object'` karena kolom tersebut tidak memiliki mean, standar deviasi, dan lain-lain. Dengan menggunakan looping kita dapat mengeluarkan nilai yang dicari dari semua kolom yang ada di data secara berurutan

## 2. Eksplorasi Data
Setelah kita mengetahui informasi statistika deskriptifnya kita sekarang dapat mencari nilai outlier, cara mencari outlier terdapat dua cara yaitu:

### Z Score
Jika kita menggunakan Z Score, Z Score sedikit kurang kebal dengan outlier, dengan arti karena Z Score didapatkan dengan menggunakan standar deviasi dan rata-rata yang dimana outlier tersebut juga ikut serta untuk menghasilkan Z Score.
```py
for i in kolom:
    if data[i].dtype == 'O':
        continue
    data[f'z_score {i}'] = stats.zscore(data[i])
    outliers_zscore = data[data[f'z_score {i}'].abs() > 3]
    display(f'Outlier z-score dari {i}',outliers_zscore)
```
Sama seperti sebelumnya kita melakukan looping untuk memisahkan terlebih dahulu kolom yang tipenya `'Object'`. Kode tersebut menggunakan fungsi yang telah ada di library scipy `stats.zscore()`

![image](https://github.com/user-attachments/assets/1d26d61b-5d2f-4b12-8707-8a4fb267e97d)

Output yang digunakan seperti digambar tersebut dan dari keelima kolom yang dilakukan pengecekan Z Score hanya kolom Horsepower yang memiliki outlier berjumlah 4 outlier

### IQR
Sedangkan jika kita menggunakan IQR, IQR sedikit lebih tahan dengan outlier daripada Z Score dikarenakan tidak menggunakan outlier untuk mencari nilai IQR
```py
for i in kolom:
    if data[i].dtype == 'O':
        continue
    Q1 = data[i].quantile(0.25)
    Q3 = data[i].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers_iqr = data[(data[i] < lower_bound) | (data[i] > upper_bound)]
    print(f'Lower bound {lower_bound} dan upper bound {upper_bound} di kolom {i}')
    display(f'Outlier dengan iqr di kolom {i}',outliers_iqr)
```
Sama seperti sebelum-sebelumnya lagi kita melakukan looping untuk memisahkan terlebih dahulu kolom yang tipenya `'Object'`. Kode tersebut menggunakan beberapa fungsi yang sudah ada didalam library pandas seperti `.quantile()` yang dimana digunakan untuk mencari quartil dari data tersebut, kemudian dari hasil quartil 3 dan quartil 1 kita kurangkan dan akan mendapatkan nilai IQR. Setelah kita mendapatkan nilai IQR kita dapat menetapkan `lower_bound`, dan `upper_bound`

![image](https://github.com/user-attachments/assets/9ec7cbb3-037b-41ed-b199-2a2e4df847fe)

Gambar diatas adalah output dari mencari outlier dengan menggunakan IQR nah disini terlihat mengapa IQR lebih tahan terhadap outlier karena dari 5 kolom terdapat 2 kolom yang teredeteksi outlier yaitu kolom horsepower(seperti z score) dan MPG

### Handling Outlier
Disini saya tidak melakukan handling outlier dikarenakan data tersebut adalah data mobil yang dalam stok di dealer, dan data di kolom MPG dibutuhkan sebagai spesifikasi mobil tersebut seperti apakah mobili ini hemat bahan bakar atau tidak?, lalu untuk kolom horsepower digunakan juga untuk spesifikasi mobil seperti contoh beberapa orang mencari mobil dengan horsepower tinggi. Dan setelah saya crosscheck kembali mobil yang tertera memang berspesifikasi sesuai dengan data

### Visualisasi
```py
kolom_numerik=[]
for i in kolom:
    if data[i].dtype == 'O':
        continue
    kolom_numerik.append(i)

fig,ax=plt.subplots(nrows=1,ncols=5,figsize=(15,8))

for i,var in enumerate(kolom_numerik):
    sns.boxplot(y=data[var], ax=ax[i])
    ax[i].set_title(f'Boxplot dari data {var}')

plt.tight_layout()
plt.show()
```
Kode tersebut diawali lagi dengan memisahkan antara kolom yang bertipe object dan angka(float/int), setelah kita `.append()` ke `kolom_numerik` kita dapat membuat dasar/tempat untuk menaruh plot yang akan kita buat dengan menggunakan `plt.subplots` dan kemudian kita kustomisasi untuk slot row kita pilih 1 row `nrows=1`, slot columns 5 karena kolom yang ingin kita box plot berjumlah 5 `ncols=5`, dan `figsize=(15,8)` untuk membuat chartnya berukuran 15*8

Setelah kita membuat subplots kita akan membuat box plotnya. Kita akan menggunakan looping untuk melakukannya secara bersamaan

- `for i, var in enumerate(kolom_numerik):` kode ini digunakan untuk memanggil setiap nilai dan setiap index dari nilai yang dipanggil contoh: `0 MPG` 0 sebagai i dan MPG sebagai var
- `sns.boxplot(y=data[var], ax=ax[i])` kode yang di loop ini berguna untuk membuat masing-masing box plot. `y=data[var]` digunakan untuk memasukkan data yang ingin dibuat box plot di sumbu y kita bisa menggunakan `x=data[var` tetapi nantinya hasil dari box plot akan menjadi horizontal, lalu fungsi dari `ax=ax[i]` digunakan untuk menempatkan boxplot yang telah dibuat di axis sublot yang kita buat diatas
- `ax[i].set_title(f'Boxplot dari data {var}')` kode ini digunakan untuk memberikan judul dari setiap boxplot yang kita buat dengan nama kolom yang sesuai

![image](https://github.com/user-attachments/assets/b4b640cd-6952-45e1-8442-ac58776996ae)

## 3 Lakukan Uji Korelasi
Uji Korelasi akan menggunakan metode pearson yang sudah tersedia di scipy

### Membuat fungsi untuk rentang korelasi (opsional)
```py
def rentang_korelasi(pearman):
    pearman=pearman[0]
    if 0.9 <= pearman <= 1.0:
        return 'Korelasi Positif Sangat kuat'
    elif -1.0 <= pearman <= -0.9:
        return 'Korelasi Negatif Sangat kuat'
    elif 0.7 <= pearman < 0.9:
        return 'Korelasi Positif kuat'
    elif -0.9 <= pearman < -0.7:
        return 'Korelasi Negatif kuat'
    elif 0.4 <= pearman < 0.7:
        return 'Korelasi Positif sedang'
    elif -0.7 <= pearman < -0.4:
        return 'Korelasi Negatif sedang'
    elif 0.1 <= pearman < 0.4:
        return "Korelasi Positif lemah"
    elif -0.4 <= pearman < -0.1:
        return 'Korelasi Negatif lemah'
    else:
        return 'Tidak ada korelasi'
```
Kode tersebut adalah fungsi yang digunakan untuk menentukan rentang korelasi dan angka untuk percabangan di kode tersebut saya dapatkan dari berselancar di internet

### Uji Korelasi
```py
hasil1=pearsonr(x=data['Horsepower'],y=data['Displacement'])
print(f"""Korelasi dari kolom horsepower dan displacement (pergerakan yang telah dilalui mobil) adalah
{rentang_korelasi(hasil1)}\n""")

hasil2=pearsonr(x=data['MPG'],y=data['Weight'])
print(f"""Korelasi dari kolom mpg (miles per galons) dan weight adalah
{rentang_korelasi(hasil2)}""")
```
Kode diatas adalah uji korelasi menggunakan `pearsonr()` kemudian hasil dari uji dari kolom horsepower dan displacement disimpan di `hasil1` lalu akan di masukkan ke fungsi, begitupun dengan perbandingan kolom MPG dengan weight

![image](https://github.com/user-attachments/assets/60ae1289-144f-4fd5-b73e-b53954142846)

> Disini saya baru tahu jika hasil dari pearsonr() adalah sebuah list jadi saya memanfaatkan pearsonr()[0] untuk menjalankan fungsi tersebut :D
