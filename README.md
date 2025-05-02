# 🚗 Car Price Prediction using K-Nearest Neighbors (KNN)

This machine learning project estimates the **resale price of used cars in Germany** using K-Nearest Neighbors Regression. The app takes four inputs from the user and predicts the price using a trained scikit-learn model — all served via a **Streamlit web app**.

---

## 📊 Features Used for Prediction

- `year`: Year of manufacture of the car
- `power_ps`: Engine power in PS (metric horsepower)
- `fuel_consumption_g_km`: Fuel consumption in grams per kilometer
- `mileage_in_km`: Total mileage driven (in kilometers)

---

## 💡 Technologies Used

| Component         | Tool/Library     |
|------------------|------------------|
| Model             | KNeighborsRegressor (scikit-learn) |
| Web App           | Streamlit        |
| Data Preprocessing| pandas, StandardScaler |
| Language          | Python 3.x       |

---

## 📁 Project Structure

Car_Price_Prediction_KNN/
├── app.py # Streamlit web app
├── knn_model.pkl # Saved KNN model
├── scaler.pkl # StandardScaler used during training
├── requirements.txt # Python dependencies
├── README.md # This documentation
├── 01Data_Cleaning.ipynb # Notebook for cleaning raw dataset
├── 02.Exploratory_Data_Analysis.ipynb
├── 03Statistical_Data_Analysis.ipynb
├── 04Feature_Engineering.ipynb
├── cars_data.csv # Raw car dataset
├── cars_dataset_cleaned.csv # Cleaned version used for training

🧠 Model Training Notes
The dataset was cleaned, analyzed, and transformed using multiple Jupyter Notebooks.

A K-Nearest Neighbors model was trained on scaled inputs using StandardScaler.

The best k was chosen through experimentation (k=5).

The trained model and scaler were saved using Python's pickle module.

