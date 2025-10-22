# Laporan Proyek Sentiment Analysis Maskapai di Twitter

## 1. Pendahuluan
Media sosial merupakan saluran utama bagi pelanggan maskapai untuk menyampaikan keluhan dan apresiasi. Penelitian ini bertujuan menganalisis sentimen pelanggan maskapai di Amerika Serikat berdasarkan dataset Tweets.csv, serta membangun model prediksi dan dashboard interaktif.

## 2. Rumusan Masalah
1. Bagaimana distribusi sentimen pelanggan terhadap masing-masing maskapai?
2. Faktor negatif apa yang paling sering muncul dan bagaimana keterkaitannya dengan maskapai tertentu?
3. Seberapa akurat model pembelajaran mesin dalam mengklasifikasikan sentimen?

## 3. Tinjauan Pustaka
- Analisis sentimen menggunakan TF-IDF dan Support Vector Machine umum digunakan pada data teks pendek seperti tweet.
- Association rule mining membantu menemukan hubungan antara entitas (maskapai) dan alasan keluhan.
- Visual analytics meningkatkan pemahaman data bagi pengambil keputusan.

## 4. Metodologi
- **Sumber Data**: Tweets.csv (14.640 entri) dengan 15 kolom utama.
- **Pra-pemrosesan**: normalisasi huruf, pembersihan URL, penghapusan stopword, dan pembentukan fitur TF-IDF bigram.
- **Analisis**: eksplorasi statistik, perhitungan distribusi sentimen, serta mining asosiasi menggunakan Apriori.
- **Modeling**: evaluasi Logistic Regression, Linear SVM, dan Complement Naive Bayes dengan split pelatihan/validasi 80/20.

## 5. Hasil dan Pembahasan
- Proporsi sentimen negatif mencapai 62.69% dari keseluruhan tweet.
- Alasan negatif terbanyak: Customer Service Issue (2910 tweet).
- Linear SVM memberikan akurasi 0.763 dan menjadi model terbaik untuk implementasi dashboard.
- Dashboard menampilkan visualisasi utama (heatmap, timeline, word cloud) dan fitur prediksi real-time.

## 6. Kesimpulan dan Saran
- Mayoritas keluhan berkaitan dengan keterlambatan penerbangan dan layanan pelanggan.
- Model klasifikasi mampu memberikan akurasi di atas target, sehingga layak digunakan untuk prediksi otomatis.
- Disarankan menambah data terbaru dan melakukan evaluasi berkala terhadap kinerja model.

## 7. Daftar Pustaka Singkat
- Liu, B. (2012). Sentiment Analysis and Opinion Mining. Morgan & Claypool.
- Bird, S., Klein, E., & Loper, E. (2009). Natural Language Processing with Python. O'Reilly.
- Han, J., Kamber, M., & Pei, J. (2011). Data Mining: Concepts and Techniques. Morgan Kaufmann.
