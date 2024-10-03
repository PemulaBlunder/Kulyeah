# Analisis Data Eksplorasi pada Dataset Pokemon Unite

1. Memuat Data  
2. Informasi Dasar  
3. Duplikat atau Nilai Unik  
4. Visualisasi Nilai Unik  
5. Menemukan Nilai Null  
6. Mengganti Nilai Null (Jika Ditemukan)  
7. Mengetahui jenis data dari dataset untuk mempermudah proses  
8. Memfilter Data  
9. Membuat Box Plot  
10. Korelasi  

## 1. Memuat Data
```py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data=pd.read_csv(C:\\Users\\Iyan\\Downloads\\old_cars.csv)
data
```
Untuk menemukan direktori file kita dapat men-select file tersebut dan akan ada simbol ... kemudian kita pilih yang copy path.

![image](https://github.com/user-attachments/assets/c42fb41f-1b09-4ec0-96da-f770362571a7)

Setelah anda memanggil data seharusnya akan mengeluarkan output seperti ini.

![image](https://github.com/user-attachments/assets/ab3cc885-0fc8-4746-af24-f0b76200cd8f)

## 2. Informasi dasar
```py
data.info()
```
Setelah anda run kode tersebut akan keluar informasi dari dataframe yang kita read.

![image](https://github.com/user-attachments/assets/0889156f-9d51-424e-ab38-4141ff8d59bc)

## 3. Duplikat atau Nilai Unik
Setelah kita mengetahui informasi-informasi dari data kita kita dapat mencari nilai duplikat.
```py
colomnya=data.columns.to_list()
for y in colomnya:
    print(fDari kolom {y} nilai duplikatnya,(data[y].duplicated().sum()))
    print(fDari kolom {y} nilai NaN nya,(data[y].isnull().sum()))
    print(fDari kolom {y} nilai uniquenya,(len(data[y].unique())))
    print(fDari kolom {y} tipe nilai nya,(data[y].dtypes),\n)
```
Disini saya menggabungkan mencari total nilai unik, mencari total duplikat, dan mencari total NaN. Menggunakan `len(data[y].unique()))` untuk mencari nilai unik, dengan menggunakan looping saya dapat melakukan beberapa hal sekaligus.

![image](https://github.com/user-attachments/assets/70cae662-2bec-4756-9ac8-cab32c4be581)

Gambar diatas adalah output dari mencari total nilai unik, mencari total duplikat, dan mencari total NaN. Dengan menggunakan looping

## 4. Visualisasi Duplikat

Setelah kita mengetahui bahwa di data kita terdapat value duplikat kita dapat memvisualisasikannya.
```py
group=data.columns.to_list()
duplicated=[]
unique_val=[]

for i in group:
    simpan_angka=0
    simpan_angka+=(len(data[i].unique()))
    unique_val.append(simpan_angka)
    duplicated.append(len(data)-simpan_angka)
```
Sebelum kita visualisasikan kita mencari berapa total duplicate dan nilai unik tiap kolom, menggunakan `len(data[i].unique())` untuk mendapatkan jumlah data yang unique dan kemudian di simpan di dalam list kosong yang telah di-inisialisasi, dan begitupun untuk nilai duplikat. 
```py
df_for_tes=pd.DataFrame({Duplicate:duplicated,Unique:unique_val})
df_for_tes.index=group
df_for_tes
```
Setelah itu kita gunakan nilai duplikat dan unik yang telah disimpan menjadi dataframe dengan kolom `Duplicate` dan nilai dari `dulpicated`, kolom `Unique` dan nilai dari `unique_val`, dan menggunakan nilai dari `group` sebagai index sehingga menjadi:

![image](https://github.com/user-attachments/assets/eb0c9bff-cd6a-46ff-b019-0669d829ff2a)

Gambar diatas adalah hasil dari kode pembuatan dataframe
```py
category=df_for_tes.index.to_list()
range_of_mpg=df_for_tes.columns.to_list()
for_chart=[]
colors=[#7c6987,#fd8c23]
for i in range_of_mpg:
    for_chart.append(df_for_tes[i])

width_bar=0.3
index=np.arange(len(category))

fig,ax=plt.subplots(figsize=(12,8))

for i in range(len(range_of_mpg)):
    ax.bar(index+i*width_bar, list(for_chart[i]), width_bar, label=f"{range_of_mpg[i]}", color=colors[i])

ax.set(xlabel=Columns,ylabel=Jumlah Data,title=Distribusi Duplicate, dan Unique
       ,xticks=index + width_bar * (len(range_of_mpg) - 1) / 2,xticklabels=category)

ax.legend()

plt.tight_layout()
plt.show()
```
`category` menyimpan indeks (label baris) dari DataFrame df_for_tes, yang mewakili nama kolom. `range_of_mpg` menyimpan nama-nama kolom dari df_for_tes, khususnya untuk "Duplicated Values" dan "Unique Values." `for_chart` diinisialisasi sebagai daftar kosong, yang nantinya akan menyimpan data untuk setiap kolom yang akan dipetakan.
`colors` adalah daftar kode warna yang digunakan untuk batang dalam grafik.

Sebuah loop bersarang mengisi `for_chart` dengan nilai dari setiap kolom `df_for_tes`, sehingga memudahkan untuk memplotnya bersama-sama.

`width_bar` didefinisikan untuk mengontrol lebar setiap batang dalam grafik.
`index` dibuat menggunakan `np.arange(len(category))` untuk menghasilkan array indeks berdasarkan jumlah kategori (panjang category).

Sebuah figura dan sumbu dibuat menggunakan `plt.subplots()` dengan ukuran yang ditentukan sebesar 12 kali 8 inci.
Loop lain iterasi melalui `range_of_mpg`, menambahkan batang untuk setiap dataset (nilai duplikat dan unik) menggunakan `ax.bar()`. `index + i * width_bar` menggeser setiap set batang agar tidak tumpang tindih.

Sumbu diberi label, dan judul ditetapkan menggunakan `ax.set()`. X-ticks diposisikan untuk memusatkan batang yang dikelompokkan dan diberi label dengan kategori.
Sebuah legenda ditambahkan untuk membedakan antara "Duplicated Values" dan "Unique Values." 
`plt.tight_layout()` dipanggil untuk menyesuaikan jarak grafik agar lebih mudah dibaca.

![image](https://github.com/user-attachments/assets/3383eb0a-d031-4e78-b4d4-74fb57e21ef0)

## 5. Menemukan Null Values
Kita telah menemukan Null Values di soal nomor 3 tadi
```py
colomnya=data.columns.to_list()
for y in colomnya:
    print(fDari kolom {y} nilai duplikatnya,(data[y].duplicated().sum()))
    print(fDari kolom {y} nilai NaN nya,(data[y].isnull().sum()))
    print(fDari kolom {y} nilai uniquenya,(len(data[y].unique())))
    print(fDari kolom {y} tipe nilai nya,(data[y].dtypes),\n)
```
Disini saya menggabungkan mencari total nilai unik, mencari total duplikat, dan mencari total NaN. Menggunakan `data[y].isnull().sum())` untuk mencari nilai NaN, dengan menggunakan looping saya dapat melakukan beberapa hal sekaligus.

![image](https://github.com/user-attachments/assets/70cae662-2bec-4756-9ac8-cab32c4be581)

## 6. Mengganti Null Values
Karena tidak ada nilai NaN di data saya maka lanjut ke pertanyaan selanjutnya

## 7. Mengetahui jenis data dari dataset untuk mempermudah proses
```py
data.info()
```
Setelah anda run kode tersebut akan keluar informasi dari dataframe yang kita read, termasuk tipe value dari masing-masing kolom.

![image](https://github.com/user-attachments/assets/0889156f-9d51-424e-ab38-4141ff8d59bc)

## 8. Memfilter Data
```py
data_japan=data[data['Origin']=='Japan']
data_japan.reset_index(drop=True,inplace=True)
display(data_japan.head())

data_us=data[data['Origin']=='US']
data_us.reset_index(drop=True,inplace=True)
display(data_us.head())

data_europe=data[data['Origin']=='Europe']
data_europe.reset_index(drop=True,inplace=True)
display(data_europe.head())
```
Kita melakukan filtering dengan menggunakan `data[data['Origin']=='Japan']` yang akan menampilkan hanya nilai di `data['Origin']` yang berisi `Japan`, dan begitupun untuk 2 Origin yang lainnya

![image](https://github.com/user-attachments/assets/01c87a11-d5c1-4006-9f98-33f5b27a2764)

## 9. Membuat Box Plot
```py
plt.figure(figsize=(10, 6))
plt.boxplot(data['MPG'], vert=False)
plt.title('Box Plot of MPG')
plt.xlabel('MPG')
plt.grid()
plt.show()
```
`plt.figure()` untuk membuat plot kosong dan `figsize=(10, 6)` untuk mengubah ukuran plot menjadi 10*6. `plt.boxplot(data['MPG'], vert=False)` untuk membuat box plot horizontal dengan value dari `data['MPG
']`

![image](https://github.com/user-attachments/assets/53fc7235-9bbb-4bec-ab93-e584e33d9e88)


## 10. Korelasi
```py
plt.figure(figsize=(12, 8))
corr_matrix = data.select_dtypes("number").corr()
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
plt.title("Heatmap of Correlation Matrix")
plt.show()
```
`plt.figure()` untuk membuat plot kosong dan `figsize=(12, 8)` untuk mengubah ukuran plot menjadi 12*8. `corr_matrix = data.select_dtypes("number").corr()` `data.select_dtypes("number")` Memilih kolom-kolom dalam DataFrame data yang memiliki tipe data numerik (misalnya, integer atau float). Ini penting karena hanya variabel numerik yang dapat dihitung korelasinya. `.corr()` Menghitung matriks korelasi untuk kolom-kolom numerik yang dipilih. Matriks korelasi menunjukkan hubungan linier antara setiap pasangan variabel numerik, dengan nilai berkisar antara -1 (korelasi negatif sempurna) hingga 1 (korelasi positif sempurna).

`sns.heatmap(corr_matrix, ...)` Menggunakan fungsi heatmap dari pustaka Seaborn untuk menggambar heatmap berdasarkan matriks korelasi yang telah dihitung. `annot=True` Menambahkan anotasi (nilai korelasi) ke setiap sel di heatmap. Ini membuatnya lebih mudah untuk membaca nilai korelasi langsung dari grafik. `cmap="coolwarm"` Menentukan skema warna untuk heatmap. coolwarm memberikan gradien warna dari biru (korelasi negatif) ke merah (korelasi positif), membantu visualisasi hubungan antara variabel.

![image](https://github.com/user-attachments/assets/91be76af-534e-43b0-b7b3-f1eb6245c1df)







