📊 Prediksi Harga Emas dengan Neural Network
Proyek ini adalah aplikasi web berbasis Flask yang digunakan untuk melakukan prediksi harga emas menggunakan model Artificial Neural Network (ANN) yang dibangun dari nol (tanpa library deep learning seperti TensorFlow/PyTorch).

🚀 Fitur Utama:
- Visualisasi Data
  Menyediakan tampilan data harga emas dalam bentuk tabel dan grafik candlestick interaktif menggunakan Plotly.
- Normalisasi Data
  Data harga emas dinormalisasi menggunakan metode Min-Max Scaling agar sesuai dengan kebutuhan input jaringan saraf.
- Split Data (Training & Testing)
  Data dibagi menjadi 70% training dan 30% testing untuk evaluasi model.
- Training Neural Network
  Model ANN dilatih dengan arsitektur sederhana (4 input neuron → 8 hidden neuron → 1 output neuron) menggunakan algoritma backpropagation.
- Evaluasi Model
  Menghitung nilai MSE (Mean Squared Error) pada data training dan testing.
- Prediksi Harga
  Pengguna dapat mengunggah file Excel dengan format kolom [Open, High, Low, Volume], dan aplikasi akan menghasilkan prediksi harga Close.

📑 Format Input Prediksi
File Excel yang diunggah harus memiliki kolom:
- Open
- High
- Low
- Volume
  
Contoh format:
Open	High	Low	Volume
1800	1820	1790	50000

🛠️ Teknologi yang Digunakan
- Python (Flask, Pandas, NumPy, Matplotlib, Plotly, Seaborn, Scikit-learn)
- HTML + Bootstrap (untuk tampilan web)
- ANN Custom Implementation (dibuat manual tanpa library deep learning)
