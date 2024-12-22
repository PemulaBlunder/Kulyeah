```markdown
# Perbandingan Metode Rekursif dan Iteratif untuk Pohon Skill

Proyek ini membandingkan metode **rekursif** dan **iteratif** untuk membuka pohon skill, sebuah fitur umum dalam gim di mana pemain membuka skill berdasarkan keterkaitan dengan skill lain.

## Gambaran Umum

Dalam banyak game, Skill Tree adalah struktur hierarkis di mana membuka sebuah skill bergantung pada penguasaan skill prasyarat. Proyek ini mengimplementasikan dua metode untuk menghitung dan membuka skill:
1. **Metode Rekursif**
2. **Metode Iteratif**

Proyek ini juga mengukur dan membandingkan kinerja kedua metode dengan menggunakan pengujian waktu eksekusi.

## Fitur

- **Implementasi Pohon Skill**: Representasi skill menggunakan struktur kelas dengan hubungan orang tua-anak.
- **Metode Rekursif dan Iteratif**: Membuka skill berdasarkan prasyarat.
- **Analisis Kinerja**: Mengukur waktu eksekusi untuk kedua metode pada berbagai iterasi.
- **Visualisasi**: Membuat grafik perbandingan waktu eksekusi.

## Memulai

### Requirement
- Python versi 3.7 atau lebih baru
- Perpustakaan yang diperlukan:
  - `matplotlib`
  - `numpy`

Pasang dependensi dengan perintah:
```bash
pip install matplotlib numpy
pip install matplotlib matplotlib
```

### File

- `gamin4.ipynb`: Script Python utama yang berisi implementasi.
- `skill_tree.json`: File JSON yang mendefinisikan struktur pohon skill.

### Contoh Struktur Pohon Skill (`skill_tree.json`)

```json
[
  {"name": "Nully", "parent_skills": []},
  {"name": "Divine Strike", "parent_skills": ["Nully"]},
  {"name": "Heaven's Grace", "parent_skills": ["Nully"]},
  {"name": "Holy Wrath", "parent_skills": ["Divine Strike", "Heaven's Grace"]}
]
```

## Cara Penggunaan

1. Clone repository ini:
   ```bash
   git clone https://github.com/username/skill-tree-comparison.git
   cd skill-tree-comparison
   ```

2. Jalankan script:
   ```bash
   python skill_tree.py
   ```

3. Amati output di konsol dan grafik perbandingan waktu eksekusi metode rekursif dan iteratif.

## Hasil Output

- **Log di Konsol**: Informasi tentang skill yang berhasil dibuka dan poin yang dibutuhkan.
- **Grafik**: Grafik perbandingan waktu eksekusi antara metode rekursif dan iteratif.

### Contoh Grafik Output

Grafik ini menunjukkan perbandingan waktu eksekusi (dalam milidetik) untuk 1.000 iterasi metode rekursif dan iteratif:

![image](https://github.com/user-attachments/assets/d8f9fdc4-5871-4218-a4aa-43c848ec846d)

## Hasil

- **Metode Rekursif**: Memiliki pendekatan yang jelas untuk menangani prasyarat skill, namun dapat mengalami batasan tumpukan (stack overflow) pada pohon skill yang sangat dalam.
- **Metode Iteratif**: Menghindari batasan rekursi dan seringkali memiliki kinerja yang lebih baik atau setara dengan metode rekursif.

## Penghargaan

- Konsep Skill Tree terinspirasi dari berbagai game RPG.
```
