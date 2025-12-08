# 🛠️ Preprocessing Tool – Smart Data Cleaning & EDA Web App  

A modern Streamlit-based data preprocessing tool designed to clean, explore, transform, and prepare messy real-world datasets for machine learning.  
This tool follows a guided, end-to-end workflow suitable for data analysts, ML engineers, and students.

---

## 🚀 Workflow  

1️⃣ Upload Dataset  
2️⃣ Fix Missing Values  
3️⃣ Semantic Cleanup  
4️⃣ Outlier Handling  
5️⃣ Exploratory Data Analysis (EDA)  
6️⃣ Encoding & Transformation  
7️⃣ Correlation Cleanup  
8️⃣ PCA (optional)  
9️⃣ Download Clean Dataset  

---

## ✨ Key Features  

- **Automatic & Manual Missing Value Handling**  
- **Semantic Cleanup:** fix mixed types, units, patterns, and inconsistencies  
- **Outlier Detection:** IQR, Z-Score, manual thresholds  
- **Skewness Correction:** log, sqrt, reciprocal  
- **Categorical Encoding:** Label, One-Hot, Manual Mapping  
- **Correlation Handling:** detect & manage highly correlated features  
- **PCA Dimensionality Reduction**  
- **EDA Auto-Charts:** histograms, boxplots, Num–Num, Num–Cat, Cat–Cat visualizations  

---

## 📸 App Overview  

### 🔧 Preprocessing Flow  
- Structured multi-page pipeline  
- Auto-detection of data types  
- Interactive visual previews  
- Before/after comparison for every step  

### 📊 Outputs  
- Clean ML-ready dataset  
- EDA summaries & visualizations  
- Encoded & transformed data  
- PCA components (optional)  

---

## ⚙ Tech Stack  

- **Streamlit** — Web app & workflow UI  
- **Pandas, NumPy** — Data manipulation  
- **scikit-learn** — Encoding, scaling, PCA  
- **Plotly** — Interactive EDA charts  

---

## ▶️ How to Run Locally  

```bash
git clone https://github.com/Techy-Tushar/Preprocessing-Tool
cd Preprocessing-Tool
pip install -r requirements.txt
streamlit run app.py
```

---

## 🛠 Future Improvements  

- Enhanced mixed-type detection & automatic normalization  
- Currency and unit conversion engine  
- Improved outlier auto-suggestions  
- Advanced EDA exports (PDF / HTML reports)  
- Heatmaps, pairplots, advanced multivariate visualizations  
- Target encoding & advanced categorical encoding techniques  
- Scree plots & PCA variance visuals  
- Performance optimization for large datasets  
- Mobile-responsive UI layout  
- Deployment on Streamlit Cloud for public demo  

---

## 📁 Project Structure  

```
Preprocessing-Tool/
│── app.py
│── requirements.txt
│── README.md
│── app_pages/
│── utils/
│── assets/        (optional)
│── models/        (optional)
```

---

## 📬 Contact  

If you find this project useful or want to collaborate, feel free to reach out!

**GitHub:** https://github.com/Techy-Tushar  
**LinkedIn:** https://www.linkedin.com/in/tusharathod/
