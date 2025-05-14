{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "9c0b3dde-042c-44fb-a83b-f70cc3916a48",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "✅ Successfully saved knn_model.joblib and scaler.joblib\n"
     ]
    }
   ],
   "source": [
    "import pickle\n",
    "import joblib\n",
    "\n",
    "# Load the model from .pkl\n",
    "with open('knn_model.pkl', 'rb') as f:\n",
    "    model = pickle.load(f)\n",
    "\n",
    "# Load the scaler from .pkl\n",
    "with open('scaler.pkl', 'rb') as f:\n",
    "    scaler = pickle.load(f)\n",
    "\n",
    "# Save as .joblib\n",
    "joblib.dump(model, 'knn_model.joblib')\n",
    "joblib.dump(scaler, 'scaler.joblib')\n",
    "\n",
    "print(\"✅ Successfully saved knn_model.joblib and scaler.joblib\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "19c08aa8-11af-4f89-b224-660e288dde53",
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
