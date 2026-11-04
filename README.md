# 🍷 Wine Quality Classification & Prediction

An end-to-end Machine Learning project that classifies and predicts the quality of wine based on its physicochemical properties using the **WineQT** dataset.

---

## 📌 Project Overview
Wine quality assessment is crucial in the beverage industry for certification, pricing, and quality control. This project explores, analyzes, and models the physicochemical properties of wine (such as acidity, residual sugar, sulfur dioxide, pH, and alcohol) to classify wine quality scores accurately using supervised machine learning algorithms.

---

## 📊 Dataset Description
The dataset used in this project is **WineQT.csv** (sourced from Kaggle / UCI Machine Learning Repository), containing **1,143 samples** and **12 input features**:

| Feature | Description |
| :--- | :--- |
| **Fixed Acidity** | Non-volatile acids present in the wine (tartaric acid) |
| **Volatile Acidity** | Amount of acetic acid in wine (too high leads to unpleasant vinegar taste) |
| **Citric Acid** | Adds 'freshness' and flavor to wine |
| **Residual Sugar** | Amount of sugar remaining after fermentation stops |
| **Chlorides** | Amount of salt in the wine |
| **Free Sulfur Dioxide** | Free form of SO₂ preventing microbial growth & oxidation |
| **Total Sulfur Dioxide** | Sum of free and bound forms of SO₂ |
| **Density** | Density of wine close to that of water depending on sugar/alcohol |
| **pH** | Level of acidity or basicity on a 0–14 scale |
| **Sulphates** | Wine additive contributing to SO₂ levels acting as an antimicrobial agent |
| **Alcohol** | Alcohol percentage content of the wine |
| **Quality** *(Target)* | Sensory score between 0 and 10 |

---

## 🛠️ Tech Stack & Libraries
- **Language:** Python
- **Data Manipulation & Analysis:** Pandas, NumPy
- **Data Visualization:** Matplotlib, Seaborn
- **Machine Learning:** Scikit-learn (Decision Tree, Random Forest, etc.)
- **Development Environment:** Jupyter Notebook / Anaconda

---

## 🚀 Workflow & Project Pipeline
1. **Exploratory Data Analysis (EDA):**
   - Checking feature distributions, missing values, and outliers.
   - Correlation heatmaps to identify strong predictors (e.g., alcohol vs. quality).
2. **Data Preprocessing:**
   - Removing redundant columns (such as `Id`).
   - Feature scaling and handling class imbalances (e.g., binary grouping: Good vs. Bad wine).
   - Splitting data into training and testing sets (`train_test_split`).
3. **Model Building & Evaluation:**
   - Training classification models (Decision Tree Classifier, Random Forest, etc.).
   - Evaluating performance using Accuracy, Precision, Recall, F1-Score, and Confusion Matrix.

---

## ⚙️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/Yadavaniket19/Wine-Quality-Classification.git](https://github.com/Yadavaniket19/Wine-Quality-Classification.git)
   cd Wine-Quality-Classification
   ```
### 2.Create a virtual environment (optional but recommended):
```bash
python -m venv env
# On Windows:
env\Scripts\activate
# On macOS/Linux:
source env/bin/activate
```

### 3.Install required dependencies:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

### 4.Run the Notebook:
```bash
jupyter notebook "wine quality prediction/wineQT.ipynb"
```

### 📈 Key Insights & Results
Alcohol content and Sulphates generally show a positive correlation with higher wine quality.

Volatile acidity shows a negative correlation; higher volatile acidity significantly degrades perceived wine quality.

👤 Author
Aniket Yadav

GitHub: @Yadavaniket19

⭐️ If you found this project helpful, please consider giving it a star!
