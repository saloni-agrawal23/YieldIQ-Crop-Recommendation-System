# 🌾YieldIQ - Decision Tree Based Agricultural Crop Recommendation System with Web Platform Integration

YieldIQ is an AI-powered web platform designed to enhance modern agriculture by providing precise crop recommendations using machine learning. With rising food demands and environmental challenges, this system aims to optimize crop production through data-driven decisions.

---

## 📌 Project Overview

In the pursuit of sustainable agriculture and improved crop yield, YieldIQ leverages machine learning techniques to recommend the most suitable crops based on various soil and environmental conditions. This system is built using a web interface to allow farmers and users to input their data and receive real-time crop suggestions.

---

## 🌿 Dataset Information

- **Source:** Crop Recommendation Dataset  
- **Total Records:** 2,200 entries  
- **Features:**
  - Nitrogen (N)
  - Phosphorus (P)
  - Potassium (K)
  - Temperature
  - Humidity
  - pH
  - Rainfall
  - Crop Label (Target variable)

Dataset preprocessing included:
- Filling missing values
- Removing duplicate records

---

## 🤖 Machine Learning Models Used

- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- Logistic Regression
- ✅ Decision Tree Classifier (Best Model)

### 🔍 Model Evaluation
- **Best Model:** Decision Tree Classifier
- **Accuracy:** 98.18%
- **Metrics Used:** Accuracy, Precision, Recall, F1-Score

---

## 💻 Tech Stack

- Python (scikit-learn, pandas, numpy)
- Flask (backend framework)
- HTML, CSS, JavaScript (frontend interface)
- Jupyter Notebook / Google Colab (model development)

---

## 🚀 Features

- User-friendly web interface for data input
- Real-time crop prediction
- Backend ML model integrated with Flask
- Simple frontend using HTML + CSS 

---

## 📂 Project Structure

YieldIQ-Crop-Recommendation-System/
│
├── Webplatform/
│ ├── app.py
│ ├── decision_tree_model.pkl
│ └── templates/
│ └── index.html
│
├── codes_analysis/
│ └── agriculture_analysis_final.py
│
├── proto codes/
│ ├── Agriculture_Analysis_Final_Proto.ipynb
│ └── Decision_Tree_Model_Training.ipynb
│
├── Crop_recommendation.csv
├── requirements.txt
├── README.md
└── .gitignore

---

## 🚀 How to Run the Project Locally

1️⃣ Clone the repository
```bash
git clone https://github.com/saloni-agrawal23/YieldIQ-Crop-Recommendation-System.git
cd YieldIQ-Crop-Recommendation-System

2️⃣ Install dependencies
pip install -r requirements.txt

3️⃣ Run the Flask application
cd Webplatform
python app.py

4️⃣ Open in browser
http://127.0.0.1:5000/
