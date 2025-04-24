### **README: Customer Churn Prediction Project**

---

## **Project Overview**
This project aims to predict **customer churn** (leaving a service) using machine learning models. Multiple models were trained and evaluated to identify the best-performing one for real-world use. Predicting churn can help businesses identify customers at risk of leaving and implement retention strategies effectively.

---

## **Project Workflow**
The project consists of the following steps:

1. **Data Analysis and Preprocessing:**
   - The dataset was analyzed, and numerical and categorical columns were identified.
   - Appropriate preprocessing steps such as scaling and encoding were applied.

2. **Training Baseline Models:**
   - Various models, including **Logistic Regression**, **Random Forest**, **Gradient Boosting**, **AdaBoost**, and **XGBoost**, were trained.
   - Initial results were compared based on overall accuracy and performance on class `0` (No Churn) and class `1` (Churn).

3. **Hyperparameter Optimization:**
   - Hyperparameters for the top three models (**Random Forest**, **Gradient Boosting**, and **XGBoost**) were optimized using **GridSearchCV**.

4. **Evaluation of Optimized Models:**
   - Optimized models were evaluated using Confusion Matrix and Classification Report.
   - The **XGBoost** model was selected as the final model due to its superior performance.

---

## **Final Model**
The **XGBoost** model, with optimized hyperparameters, demonstrated the best performance and was chosen as the final model.  
- **Overall Accuracy:** 92%  
- **Precision (Class `1`):** 83%  
- **Recall (Class `1`):** 88%  
- **F1-Score (Class `1`):** 86%  

The final model has been saved and is ready for deployment to predict new data.

---

## **Requirements**
To run this project, the following dependencies need to be installed:
- **Python 3.8+**
- **Required Libraries:**
  ```bash
  pip install pandas numpy scikit-learn matplotlib seaborn xgboost joblib
  ```

---

## **How to Use**
### **1. Run the Code:**
Execute the main script (`main.py`) to train, evaluate, and save the final model.

### **2. Save and Load the Final Model:**
To save the final model:
```python
import joblib
joblib.dump(optimized_models['XGBoost'], 'final_xgboost_model.pkl')
```

To load the saved model:
```python
model = joblib.load('final_xgboost_model.pkl')
```

### **3. Predict New Data:**
To make predictions on new data:
```python
# New data for prediction (after preprocessing)
new_data = [[...]]  # Preprocessed new data

# Predict using the model
prediction = model.predict(new_data)
print("Prediction:", prediction)
```

---

## **Model Evaluation**
### **Random Forest:**
- Overall Accuracy: 92%
- Precision (Class `1`): 84%
- Recall (Class `1`): 86%

### **Gradient Boosting:**
- Overall Accuracy: 91%
- Precision (Class `1`): 82%
- Recall (Class `1`): 87%

### **XGBoost (Final Model):**
- Overall Accuracy: 92%
- Precision (Class `1`): 83%
- Recall (Class `1`): 88%
- F1-Score (Class `1`): 86%

---

## **File Structure**
- **main.py**: The main script containing preprocessing, model training, optimization, and evaluation code.
- **final_xgboost_model.pkl**: The saved final model using XGBoost.
- **README.md**: Project documentation (this file).

---

## **Conclusion**
The **XGBoost** model was chosen as the final model due to its better performance on the test data and balanced metrics. This model is now ready for deployment to predict customer churn and help businesses with retention strategies.

---
