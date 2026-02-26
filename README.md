# 🌾 YieldIQ – Decision Tree Based Agricultural Crop Recommendation System

**YieldIQ** is a machine learning–powered crop recommendation system designed to support modern agriculture through data-driven decision-making.  
The system predicts the most suitable crop based on soil nutrients and environmental conditions, helping improve crop yield and sustainability.

---

## 📌 Project Overview

With increasing food demand and environmental challenges, selecting the right crop for specific soil and climate conditions is crucial.  
YieldIQ leverages machine learning techniques to analyze agricultural data and recommend optimal crops through a simple, user-friendly web interface.

---

## 🌿 Dataset Information

- **Dataset:** Crop Recommendation Dataset  
- **Total Records:** 2,200  
- **Features:**
  - Nitrogen (N)
  - Phosphorus (P)
  - Potassium (K)
  - Temperature
  - Humidity
  - Soil pH
  - Rainfall
  - Crop Label (Target Variable)

### Preprocessing Steps
- Removal of duplicate records  
- Handling missing values

---

## 🤖 Machine Learning Models Used

The following models were trained and evaluated:

- Logistic Regression  
- Support Vector Machine (SVM)  
- K-Nearest Neighbors (KNN)  
- ✅ **Decision Tree Classifier (Selected Model)**  

### 🔍 Model Evaluation
- **Best Model:** Decision Tree Classifier  
- **Accuracy:** **98.18%**  
- **Evaluation Metrics:** Accuracy, Precision, Recall, F1-Score  

The Decision Tree model was selected due to its high accuracy and interpretability.

---

## 💻 Tech Stack

- **Programming Language:** Python  
- **Machine Learning:** scikit-learn  
- **Data Processing:** Pandas, NumPy  
- **Visualization:** Matplotlib, Seaborn, Plotly  
- **Backend:** Flask  
- **Frontend:** HTML, CSS  
- **Development Tools:** Jupyter Notebook / Google Colab  

---

## 🚀 Features

- Intuitive web interface for user input  
- Real-time crop prediction using a trained ML model  
- Flask-based backend integration  
- Lightweight and responsive frontend  

---
## 📂 Project Structure

```
YieldIQ-Crop-Recommendation-System/
│
├── Webplatform/
│   ├── app.py
│   ├── decision_tree_model.pkl
│   └── templates/
│       └── index.html
│
├── codes_analysis/
│   └── agriculture_analysis_final.py
│
├── proto codes/
│   ├── Agriculture_Analysis_Final_Proto.ipynb
│   └── Decision_Tree_Model_Training.ipynb
│
├── screenshots/
│   ├── form.png
│   └── result.png
│
├── Crop_recommendation.csv
├── workflow.svg
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

## 📊 Results

The Decision Tree model achieved high accuracy across multiple crop classes

The system provides reliable crop recommendations based on real-world parameters

The complete machine learning pipeline was successfully deployed using Flask

---

## 🔮 Future Enhancements

Deploy the application on cloud platforms (Render / AWS)

Integrate fertilizer recommendation functionality

Support regional language interfaces for farmers

---

## 👤 Author

Saloni Agrawal
GitHub: https://github.com/saloni-agrawal23

---

## 📜 License

This project is licensed under the MIT License.
