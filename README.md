# Malware Scan Demo Pipeline

A comprehensive machine learning pipeline and web application designed to classify malware families using a hybrid approach of deep learning (ResNet-18) and traditional machine learning models (Random Forest, SVM, k-NN).

## 🚀 Features
- **Deep Learning Classifier**: Custom PyTorch CNN (ResNet-18) for classifying malware families from binaries converted to images.
- **Traditional ML Models**: Scikit-learn based implementations of Random Forest, SVM, and k-NN.
- **Feature Extraction**: Robust extraction pipeline for binary analysis.
- **Flask API**: Lightweight backend API to serve model predictions.
- **React/Vite Frontend**: Modern, responsive drag-and-drop web interface for easy file uploads and model selection.
- **End-to-End Evaluation**: Automated scripts to generate dataset splits, train models, evaluate accuracy, and generate plots.

---

## 📊 Dataset Requirements

**Important:** The raw malware binaries and datasets are **not** included in this repository due to GitHub's file size limits and safety policies regarding malware samples. 

To run this project, you must download the dataset separately:
1. Download the dataset from https://www.kaggle.com/datasets/manmandes/malimg.
2. Extract the contents.
3. Place the raw dataset into the `data/raw/` directory.

---

## 🛠️ Tech Stack
- **Backend/ML:** Python, PyTorch, Scikit-learn, Flask, Pandas, OpenCV
- **Frontend:** React, Vite, CSS
- **Tools:** Joblib (model saving), matplotlib/seaborn (visualization)

---

## ⚙️ Prerequisites
Ensure you have the following installed on your system:
- Python 3.9+
- Node.js 18+ and npm

---

## 💻 Installation & Setup

### 1. Backend Setup
Clone the repository and set up your Python virtual environment:
```bash
git clone https://github.com/yourusername/malware-scan-demo.git
cd malware-scan-demo

# Create and activate a virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Mac/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Frontend Setup
Open a new terminal, navigate to the `frontend` folder, and install Node dependencies:
```bash
cd frontend
npm install
```

---

## 🏃‍♂️ How to Run

### 1. Training the Models
Before running the API, you must generate the dataset splits and train the models. Ensure your raw data is placed in `data/raw/`.

Run the master pipeline script from the root directory:
```bash
python main.py
```
This script will sequentially:
1. Build dataset splits (`data/splits/`).
2. Extract features.
3. Train the CNN (`outputs/models/resnet18_best.pth`).
4. Train ML models (`outputs/models/rf_model.pkl`, etc.).
5. Run evaluations and generate plots (`outputs/plots/`).

*Note: You can skip specific steps using flags like `--skip-convert` or `--skip-cnn`.*

### 2. Starting the API Server
Once models are trained, start the Flask API:
```bash
python app.py
```
The server will run at `http://127.0.0.1:5001`.

### 3. Starting the Frontend UI
In a separate terminal, run the Vite development server:
```bash
cd frontend
npm run dev
```
Open your browser to the local address provided by Vite (usually `http://localhost:5173`) to use the interface.

---

## 📂 Project Structure
```text
.
├── app.py                 # Flask API backend
├── main.py                # Full pipeline runner (train & evaluate)
├── predict.py             # CLI inference script
├── requirements.txt       # Python dependencies
├── src/                   # Core Python modules (model, train, evaluate, etc.)
├── frontend/              # React frontend source code
├── data/                  # Ignored: Dataset directory (raw and splits)
└── outputs/               # Ignored: Model weights, results, and plots
```

---

## ⚠️ Known Limitations
- The current implementation is a **malware-family classifier**, meaning it predicts which family a file belongs to. It does not output a strictly "benign vs. malware" verdict without appropriate benign samples.
- Executables are converted to images for the CNN; performance heavily depends on the conversion algorithm and resolution.

---

## 📜 License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
