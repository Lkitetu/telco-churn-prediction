📱 Telco Customer Churn Prediction
**ISOM 835 Individual Term Project - Predictive Analytics and Machine Learning**

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Status](https://img.shields.io/badge/Status-Complete-success.svg)
![License](https://img.shields.io/badge/License-Academic-green.svg)

## 📊 Project Overview

This project applies machine learning techniques to predict customer churn for a telecommunications company. By identifying customers at high risk of leaving, the company can implement targeted retention strategies and reduce revenue loss.

**Key Question:** Can we accurately predict which customers will churn based on their service usage patterns, contract details, and demographic information?

### Business Impact
- **Problem:** Customer churn costs telecom companies millions in lost revenue
- **Solution:** Predictive model to identify at-risk customers before they leave
- **Value:** Enable proactive retention campaigns with 80%+ accuracy and 0.846 ROC-AUC

---

## 🎯 Project Objectives

1. Build predictive models to classify customers as likely to churn or stay
2. Compare multiple machine learning algorithms (Logistic Regression, Random Forest, Gradient Boosting)
3. Identify key factors driving customer churn
4. Provide actionable business recommendations for customer retention
5. Evaluate ethical considerations in customer churn prediction

---

## 📁 Dataset

**Source:** [Kaggle - Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

**Size:** 7,043 customers × 21 features

**Target Variable:** Churn (Yes/No)

### Key Features
- **Customer Demographics:** Gender, senior citizen status, partners, dependents
- **Service Information:** Phone service, internet service, streaming services
- **Account Details:** Contract type, payment method, billing preferences
- **Usage Metrics:** Tenure (months), monthly charges, total charges

**Churn Rate:** ~26.5% (imbalanced classification problem)

---

## 🛠️ Tools & Technologies

### Programming & Environment
- **Language:** Python 3.8+
- **Platform:** Google Colab
- **Version Control:** Git/GitHub

### Libraries
```python
# Data Analysis
pandas, numpy

# Visualization
matplotlib, seaborn

# Machine Learning
scikit-learn (LogisticRegression, RandomForestClassifier, GradientBoostingClassifier)

# Evaluation
sklearn.metrics (accuracy_score, roc_auc_score, confusion_matrix, classification_report)
```

---

## 📓 Analysis Notebook

### **[View Complete Analysis on Google Colab](https://colab.research.google.com/drive/1eYi20RL6z9L910JxJMPoV-XCu3mXJ9fS?usp=sharing<img width="1388" height="1414" alt="image" src="https://github.com/user-attachments/assets/04c890de-1b1b-4860-b474-8c50f0d87bdf" />
)**

This comprehensive Jupyter notebook includes:
1. **Exploratory Data Analysis** with 6+ visualizations
2. **Data preprocessing** and feature engineering
3. **Three machine learning models** with detailed comparison
4. **Feature importance analysis**
5. **Business insights** and actionable recommendations
6. **Ethics and Responsible AI** considerations

**To view the notebook:**
1. Click the Colab link above
2. The notebook is view-only (no setup required)
3. All code and results are included

---

## 🔍 Methodology

### 1. Exploratory Data Analysis
- Analyzed distribution of 21 features across 7,043 customers
- Examined churn patterns by customer segments (contract type, tenure, pricing)
- Created visualizations: histograms, correlation heatmaps, box plots
- Identified data quality issues (11 missing values in TotalCharges)

### 2. Data Preprocessing
- Handled missing values using median imputation
- Encoded categorical variables using Label Encoding
- Feature engineering: Created `ChargesPerMonth` and `TenureGroup` features
- Split data: 80% training (5,634 customers), 20% testing (1,409 customers) with stratification
- Scaled numerical features using StandardScaler

### 3. Model Development
Implemented and compared three algorithms:

| Model | Type | Key Characteristics |
|-------|------|---------------------|
| **Logistic Regression** | Linear | Baseline model, interpretable, fast training |
| **Random Forest** | Ensemble | Handles non-linearity, provides feature importance |
| **Gradient Boosting** | Ensemble | Sequential learning, often highest performance |

### 4. Evaluation Metrics
- **Accuracy:** Overall prediction correctness
- **ROC-AUC:** Model's ability to distinguish between churners and non-churners
- **Confusion Matrix:** Breakdown of True/False Positives/Negatives
- **Precision/Recall:** Performance on positive class (churners)

---

## 📈 Key Results

### Model Performance Comparison

| Model | Accuracy | ROC-AUC | Notes |
|-------|----------|---------|-------|
| **Logistic Regression** ⭐ | **80.41%** | **0.8459** | Best overall performance |
| Gradient Boosting | 80.13% | 0.8431 | Close second |
| Random Forest | 77.43% | 0.8208 | Good feature importance insights |

**Best Model:** Logistic Regression
- Achieved highest ROC-AUC despite being the simplest model
- Demonstrates that linear relationships effectively captured churn patterns
- Superior generalization without overfitting

### Top Churn Predictors
1. **Contract Type** - Month-to-month contracts churn at 42% vs. 3% for 2-year contracts
2. **Tenure** - New customers (<12 months) are at highest risk
3. **Monthly Charges** - Higher charges correlate with increased churn
4. **Total Charges** - Cumulative spend reflects customer investment
5. **Internet Service Type** - Fiber optic customers show different patterns

---

## 💡 Business Recommendations

### 1. **Targeted Retention Campaigns**
- **Action:** Score all customers monthly, target top 10% high-risk (>70% churn probability)
- **Target:** ~400-500 customers monthly
- **Expected Impact:** Retain 15-20% of targeted segment
- **ROI:** Estimated 3-5x return on retention spend

### 2. **Contract Optimization Program**
- **Finding:** Month-to-month contracts have 13x higher churn rate
- **Action:** Incentivize 1-year and 2-year contract upgrades
- **Tactics:** Discounts, waived fees, service bundles
- **Goal:** Reduce month-to-month base by 20%

### 3. **New Customer Onboarding**
- **Finding:** First 12 months are highest-risk period
- **Action:** Enhanced onboarding with 30-60-90 day check-ins
- **Goal:** Reduce first-year churn by 25%

### 4. **Pricing Strategy Review**
- **Finding:** Monthly charges >$80 correlate with higher churn
- **Action:** Review value proposition for high-charge customers
- **Consideration:** Service bundling, competitive benchmarking

### 5. **Payment Method Modernization**
- **Finding:** Electronic check users churn more
- **Action:** Promote auto-pay with incentives
- **Dual Benefit:** Lower churn + operational efficiency

### Expected Business Impact
- **Revenue Protection:** Retain 15-20% of at-risk customers
- **Financial Impact:** ~$4.8M annually in protected revenue
- **Churn Reduction:** 3-5 percentage point decrease overall

---

## ⚖️ Ethics & Responsible AI

### Fairness Considerations
- Monitor predictions across demographic segments (age, gender)
- Ensure equal access to retention offers
- Regular fairness audits to detect bias

### Privacy & Security
- Customer data anonymized in analysis
- Predictions stored securely with access controls
- Compliance with GDPR/CCPA requirements

### Transparency
- Feature importance explains model decisions
- Human oversight on retention decisions
- Customers can opt-out of targeted marketing

### Responsible Deployment
- Quarterly fairness audits across customer segments
- Model retraining every 6 months with fresh data
- Performance monitoring to detect drift

---

## 📂 Repository Structure

```
telco-churn-prediction/
│
├── README.md                          # Project overview (this file)
├── telco_churn_ENHANCED.ipynb        # Complete analysis notebook
│
├── visualizations/                    # Generated plots and charts
│   ├── churn_distribution.png
│   ├── eda_comprehensive.png
│   ├── model_comparison.png
│   ├── confusion_matrices.png
│   ├── roc_curves.png
│   └── feature_importance.png
│
└── report/
    └── Ngumba_Leon_ISOM835_Project.pdf  # Final written report
```

---

## 🚀 How to Use This Project

### View the Analysis
1. Click the [Google Colab link](#analysis-notebook) above
2. All code, visualizations, and results are included
3. No setup required - just view and explore!

### Replicate the Analysis (Optional)
If you want to run the code yourself:

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/telco-churn-prediction.git
cd telco-churn-prediction

# Open in Google Colab or Jupyter Notebook
# Upload the dataset from Kaggle (link in notebook)
# Run all cells
```

### Requirements
- Python 3.8+
- Libraries: pandas, numpy, matplotlib, seaborn, scikit-learn
- Dataset: Available on Kaggle (link in notebook)

---

## 📊 Sample Visualizations

### Churn Distribution
Shows the class imbalance in the dataset (73% No Churn, 27% Churn)

### Model Comparison
Bar charts comparing Accuracy and ROC-AUC across all three models

### ROC Curves
Demonstrates model discrimination ability - all models significantly outperform random guessing

### Feature Importance
Top 15 features ranked by importance in predicting churn

*(All visualizations available in the `visualizations/` folder)*

---

## 🎓 Academic Context

**Course:** ISOM 835 - Predictive Analytics and Machine Learning  
**Institution:** Suffolk University - Sawyer Business School  
**Instructor:** Prof. Hasan Arslan  
**Semester:** Fall 2024  
**Submission Date:** December 12, 2024

---

## 📚 References

### Dataset
- IBM Sample Data Sets. (2020). *Telco Customer Churn*. Kaggle.  
  https://www.kaggle.com/datasets/blastchar/telco-customer-churn

### Technical Documentation
- Pedregosa et al. (2011). *Scikit-learn: Machine Learning in Python*. JMLR.  
  https://scikit-learn.org/
- McKinney, W. (2010). *Data Structures for Statistical Computing in Python*.
- Waskom, M. (2021). *Seaborn: Statistical Data Visualization*.

### Learning Resources
- Géron, A. (2019). *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow* (2nd ed.). O'Reilly Media.
- James, G., Witten, D., Hastie, T., & Tibshirani, R. (2013). *An Introduction to Statistical Learning*. Springer.

---

## 👤 Author

**Leon Ngumba**  
Graduate Student, MS Business Analytics  
Suffolk University - Sawyer Business School  
Graduate Reporting Data and Analysis Fellow

---

## 🙏 Acknowledgments

- **Prof. Hasan Arslan** - Course instruction and project guidance
- **ISOM 835 Classmates** - Collaborative learning and support
- **Suffolk University** - Resources and academic environment
- **AI Assistance:** Claude (Anthropic) - Code debugging, documentation structure, README template
  - *Note: All analysis, model development, and insights are original work*

---

## 📝 License

This project is for academic purposes as part of ISOM 835 coursework at Suffolk University.

---

**⭐ If you found this project helpful or interesting, please consider starring the repository!**

*Last Updated: December 12, 2024*

---

## 📧 Questions or Feedback?

Feel free to reach out if you have questions about the methodology, business applications, or replicating this analysis.

**Happy to discuss data science, machine learning, or customer analytics!** 🚀
