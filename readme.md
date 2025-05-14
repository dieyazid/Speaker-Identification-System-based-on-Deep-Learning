# 🎙️ Speaker Identification System using Deep Learning

This project is a deep learning-based speaker identification system that supports training, pruning (optimization), and inference (prediction) for both single and multiple speakers. It was developed as the thesis project for my Master’s degree and tested successfully on **Python 3.12.3**.

> ⚠️ **Note:** This project works with Python 3.12.3. Compatibility with future Python versions is not guaranteed.

---

## 📁 Project Structure

```
.
├── Dataset/
│   ├── RawData-set/
│   ├── prepare_dataset.py
├── Models/
│   ├── pruning.py
├── train.py
├── predict.py
├── multi_predict.py
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/dieyazid/Speaker-Identification-System-based-on-Deep-Learning.git
cd Speaker-Identification-System-based-on-Deep-Learning
```

### 2. Create and activate a virtual environment (recommended)

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 📊 Data Preparation

### If the dataset is missing or empty:

1. Copy your dataset into:

```bash
Dataset/RawData-set/
```

2. Then run:

```bash
cd Dataset
python prepare_dataset.py
cd ..
```

---

## 🧠 Model Training

Train the model by running:

```bash
python train.py
```

---

## 🪚 Model Optimization (Pruning)

To prune and optimize the trained model for deployment:

```bash
cd Models
python pruning.py
cd ..
```

> ⚠️ **Note:** Pruning is currently not functioning as expected. Troubleshooting and updates may be required.

---

## 🔍 Inference: Using the Model

### Single Audio File Prediction

```bash
# Predict speaker from a single audio file
python predict.py -m single path_to_audio_file.wav

# Predict speaker from multiple voices in the same audio
python predict.py -m multi path_to_audio_file.wav

# Record audio from microphone and predict
python predict.py -m record
```

### Multi-Speaker Prediction from Dataset

```bash
python multi_predict.py
```

---

## 💬 Notes

- This project uses TensorFlow and other common machine learning libraries.
- Performance may vary depending on system hardware and audio quality.
- You can improve accuracy by expanding the dataset and retraining.
- Pruning may need review and debugging.

---

## 📜 License

This project is for academic and research purposes. Feel free to explore, modify, and build upon it.

---

## 🙋‍♂️ Author

**Yazid**  
Master’s in AI & Machine Learning  
📬 Telegram: [@dieyazid](https://t.me/dieyazid)