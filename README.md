# 🛠️ Preprocessing Tool – Smart Data Cleaning & EDA Web App

A powerful **Streamlit-based data preprocessing application** designed to clean, transform, analyze, and prepare real-world datasets for machine learning.  
This tool automates messy data cleaning tasks and provides a smooth, guided workflow for analysts and ML practitioners.

---

## 🚀 Features

### 🧹 1. Missing Value Handler
- Automatic and manual imputation  
- Smart suggestions  
- Mean / Median / Mode / Zero / Custom value  
- Drop-row option  
- Before/After preview  
- Undo support  

---

### 🧠 2. Semantic Cleanup
Fixes hidden data-quality issues:
- Mixed data types  
- Unit standardization  
- Numeric extraction from dirty text  
- Date parsing  
- Phone/ID cleanup  
- Pattern fixing  
- String normalization  

---

### 📉 3. Outlier Detection & Fixing
- IQR, Z-Score, Manual thresholds  
- Before/After visualization  
- Outlier % report  
- Capping, removal, or replacement  
- Undo support  

---

### 📊 4. EDA Core
Automatically generates:
- Histograms  
- Boxplots  
- Scatter plots  
- Bar charts  
- Correlation heatmap  
- Insight cards  

---

### 📦 5. EDA Export Center
- Select multiple plots  
- Export full EDA report  
- JSON summary  
- Chart pack downloads  

---

### 🔢 6. Encoding & Transformation
- Label Encoding  
- One-Hot Encoding  
- Manual Mapping  
- Skewness correction (Log, SQRT, Reciprocal)  
- Correlation handling  
- PCA (2D & 3D preview)  

---

### 📥 7. Download Center
- Export cleaned dataset  
- Export interim pipeline datasets  
- Export EDA summary  
- Download pipeline actions  

---

## 🧱 Project Structure

```
CP2Project/
│
├── app.py
├── requirements.txt
├── .gitignore
│
├── app_pages/
│   ├── p1_Data_Explorer.py
│   ├── p2_Fix_Missing_Values.py
│   ├── p2b_Fix_Semantic_Cleanup.py
│   ├── p3_Outlier_Handling.py
│   ├── p4_EDA_Core.py
│   ├── p4b_EDA_Exports.py
│   ├── p5_Encoding_and_Transformation.py
│   └── p6_Download_Center.py
│
├── utils/
│   ├── theme.py
│   └── state_helpers.py
│
├── assets/
├── models/
└── venv/
```

---

## 🛠️ Tech Stack

| Component | Technology |
|----------|------------|
| UI Framework | Streamlit |
| Data Handling | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Visualization | Plotly, Matplotlib, Seaborn |
| Excel Support | openpyxl, xlrd |
| Date Parsing | python-dateutil |

---

## 📦 Installation

### 1️⃣ Clone the repository:

```bash
git clone <your_repo_link>
cd CP2Project
```

### 2️⃣ Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows
```

### 3️⃣ Install dependencies:

```bash
pip install -r requirements.txt
```

### 4️⃣ Run the app:

```bash
streamlit run app.py
```

---

## 👨‍💻 Author

**Tushar Rathod**  
Data Analyst • ML Practitioner  

- GitHub: https://github.com/Techy-Tushar  
- LinkedIn: https://www.linkedin.com/in/tusharathod  

---

## ⭐ If you like this project, please give it a star!


