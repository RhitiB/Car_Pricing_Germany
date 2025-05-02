{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "80c05d6a-49e7-47f4-99ee-fda761474ab8",
   "metadata": {},
   "source": [
    "# Car Price Prediction in the German Used Market"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ec87e579-25f8-4e4f-9d48-e5c0110fd34c",
   "metadata": {},
   "source": [
    "## 1. 📌 Introduction\n",
    "- Objective: To deploy the model"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "104ce946-48f8-4cac-b27a-5d1914e2b166",
   "metadata": {},
   "source": [
    "From Feature selection we came to know that KNN Model is the best model for car prediction. Let us create the model once more for deployment. Let us import the basic modules for training and deploying the model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "id": "e8916f41-dc37-4647-8d62-d99cf9f90baf",
   "metadata": {},
   "outputs": [],
   "source": [
    "# BASIC MODULES\n",
    "import pandas as pd\n",
    "from pprint import pprint\n",
    "import pickle\n",
    "import pandas as pd\n",
    "import streamlit as st\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "from sklearn.neighbors import KNeighborsRegressor\n",
    "from sklearn.model_selection import train_test_split"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "2c5bdd45-23ee-45b6-bd2a-59345eb0a799",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>brand</th>\n",
       "      <th>model</th>\n",
       "      <th>color</th>\n",
       "      <th>registration_date</th>\n",
       "      <th>year</th>\n",
       "      <th>price_in_euro</th>\n",
       "      <th>power_kw</th>\n",
       "      <th>power_ps</th>\n",
       "      <th>transmission_type</th>\n",
       "      <th>fuel_type</th>\n",
       "      <th>fuel_consumption_l_100km</th>\n",
       "      <th>fuel_consumption_g_km</th>\n",
       "      <th>mileage_in_km</th>\n",
       "      <th>offer_description</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>alfa-romeo</td>\n",
       "      <td>Alfa Romeo GTV</td>\n",
       "      <td>red</td>\n",
       "      <td>01.10.1995</td>\n",
       "      <td>1995.0</td>\n",
       "      <td>1300.0</td>\n",
       "      <td>148.0</td>\n",
       "      <td>201.0</td>\n",
       "      <td>Manual</td>\n",
       "      <td>Petrol</td>\n",
       "      <td>10.0</td>\n",
       "      <td>260.00</td>\n",
       "      <td>1605000.0</td>\n",
       "      <td>2.0 V6 TB</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>alfa-romeo</td>\n",
       "      <td>Alfa Romeo Spider</td>\n",
       "      <td>black</td>\n",
       "      <td>01.07.1995</td>\n",
       "      <td>1995.0</td>\n",
       "      <td>4900.0</td>\n",
       "      <td>110.0</td>\n",
       "      <td>150.0</td>\n",
       "      <td>Manual</td>\n",
       "      <td>Petrol</td>\n",
       "      <td>9.0</td>\n",
       "      <td>225.00</td>\n",
       "      <td>1895000.0</td>\n",
       "      <td>2.0 16V Twin Spark L</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>alfa-romeo</td>\n",
       "      <td>Alfa Romeo 164</td>\n",
       "      <td>red</td>\n",
       "      <td>01.11.1996</td>\n",
       "      <td>1996.0</td>\n",
       "      <td>17950.0</td>\n",
       "      <td>132.0</td>\n",
       "      <td>179.0</td>\n",
       "      <td>Manual</td>\n",
       "      <td>Petrol</td>\n",
       "      <td>7.0</td>\n",
       "      <td>52.15</td>\n",
       "      <td>961270.0</td>\n",
       "      <td>3.0i Super V6, absoluter Topzustand !</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "        brand              model  color registration_date    year  \\\n",
       "0  alfa-romeo     Alfa Romeo GTV    red        01.10.1995  1995.0   \n",
       "1  alfa-romeo  Alfa Romeo Spider  black        01.07.1995  1995.0   \n",
       "2  alfa-romeo     Alfa Romeo 164    red        01.11.1996  1996.0   \n",
       "\n",
       "   price_in_euro  power_kw  power_ps transmission_type fuel_type  \\\n",
       "0         1300.0     148.0     201.0            Manual    Petrol   \n",
       "1         4900.0     110.0     150.0            Manual    Petrol   \n",
       "2        17950.0     132.0     179.0            Manual    Petrol   \n",
       "\n",
       "   fuel_consumption_l_100km  fuel_consumption_g_km  mileage_in_km  \\\n",
       "0                      10.0                 260.00      1605000.0   \n",
       "1                       9.0                 225.00      1895000.0   \n",
       "2                       7.0                  52.15       961270.0   \n",
       "\n",
       "                       offer_description  \n",
       "0                              2.0 V6 TB  \n",
       "1                   2.0 16V Twin Spark L  \n",
       "2  3.0i Super V6, absoluter Topzustand !  "
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df_main=pd.read_csv(\"cars_dataset_cleaned.csv\")\n",
    "df_main.head(3)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "c30fadec-7996-4c64-b49e-173503d82270",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "{'algorithm': 'auto',\n",
      " 'leaf_size': 30,\n",
      " 'metric': 'minkowski',\n",
      " 'metric_params': None,\n",
      " 'n_jobs': None,\n",
      " 'n_neighbors': 5,\n",
      " 'p': 2,\n",
      " 'weights': 'uniform'}\n"
     ]
    }
   ],
   "source": [
    "features=[\"year\", \"power_ps\", \"fuel_consumption_g_km\", \"mileage_in_km\"]\n",
    "target = 'price_in_euro'\n",
    "X = df_main[features].dropna()\n",
    "y = df_main.loc[X.index, target]\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n",
    "scaler = StandardScaler()\n",
    "X_train_scaled = scaler.fit_transform(X_train)\n",
    "knn_model = KNeighborsRegressor(n_neighbors=5)\n",
    "knn_model.fit(X_train_scaled, y_train)\n",
    "pprint(knn_model.get_params())"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "6be48e15-cdb7-4e52-8d19-2af73e4509a2",
   "metadata": {},
   "source": [
    "## Deployment of the Model"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "09afaec0-c9e6-4597-8b02-569f61db43cb",
   "metadata": {},
   "source": [
    "### Step 1. Saving the trained KNN Model and Scaler\n",
    "\n",
    "This allows us to reuse the model later without retraining"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "id": "1b956db6-db0c-44be-8f36-41b1ee9927e3",
   "metadata": {},
   "outputs": [],
   "source": [
    "with open (\"knn_model.pkl\", \"wb\") as f:\n",
    "    pickle.dump(knn_model,f)\n",
    "\n",
    "# Saving the scalar used to scale the output\n",
    "\n",
    "with open (\"scaler.pkl\", \"wb\") as f:\n",
    "    pickle.dump(scaler, f)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "3648456d-3330-4e23-ac8e-a66cb18fbd30",
   "metadata": {},
   "source": [
    "This will save\n",
    "- knn_model.pkl $\\rightarrow$ as the trained model\n",
    "- scaler.pkl $\\rightarrow$ as the fitted StandardScaler"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "81e2b321-1ba8-4cc3-be70-8a2f5ab9d5b4",
   "metadata": {},
   "source": [
    "### Step 2. Creating a Simple Streamlit App"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "8902c73f-06e1-43fa-a800-087feb7b6727",
   "metadata": {},
   "source": [
    "This web app lets users input car features and get a price prediction from your saved KNN model."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "id": "d611b8b3-1766-403a-9474-4ab7a8208733",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2025-05-02 13:35:27.348 \n",
      "  \u001b[33m\u001b[1mWarning:\u001b[0m to view this Streamlit app on a browser, run it with the following\n",
      "  command:\n",
      "\n",
      "    streamlit run C:\\Users\\shils\\anaconda3\\Lib\\site-packages\\ipykernel_launcher.py [ARGUMENTS]\n",
      "2025-05-02 13:35:27.349 Session state does not function when running a script without `streamlit run`\n"
     ]
    }
   ],
   "source": [
    "with open('knn_model.pkl', 'rb') as f:\n",
    "    knn_model = pickle.load(f)\n",
    "with open('scaler.pkl', 'rb') as f:\n",
    "    scaler = pickle.load(f)\n",
    "st.title(\"🚗 Car Price Prediction (KNN Model)\")\n",
    "mileage = st.number_input(\"Mileage (in km)\", min_value=0, max_value=500000, step=1000)\n",
    "power = st.number_input(\"Engine Power (in PS)\", min_value=10, max_value=1000, step=10)\n",
    "age = st.number_input(\"Age of Car (in years)\", min_value=0, max_value=30, step=1)\n",
    "if st.button(\"Predict Price\"):\n",
    "    # Prepare and scale input\n",
    "    user_input = pd.DataFrame({\n",
    "        'mileage_in_km': [mileage],\n",
    "        'power_ps': [power],\n",
    "        'age': [age]\n",
    "    })\n",
    "\n",
    "    user_scaled = scaler.transform(user_input)    \n",
    "    price = knn_model.predict(user_scaled)[0]\n",
    "    st.success(f\"💰 Estimated Resale Price: €{price:,.2f}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "c6da36ee-a97a-4a2e-915b-5802285fa370",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python [conda env:base] *",
   "language": "python",
   "name": "conda-base-py"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
