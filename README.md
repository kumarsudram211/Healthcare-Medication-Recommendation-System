# Healthcare-Medication-Recommendation-System
# 🧠 Disease Prediction & Medicine Recommendation System

This project is a **Streamlit-based web application** that predicts diseases based on user-entered symptoms and provides a full medical recommendation suite including:

- ✅ Disease description
- 🍽️ Diet recommendations
- 💊 Medications
- ⚠️ Precautions
- 🏃 Workout plans

---

## 📁 Dataset

The dataset used in this project is sourced from Kaggle whose link is following:  
🔗 [Medicine Recommendation System Dataset](https://www.kaggle.com/datasets/noorsaeed/medicine-recommendation-system-dataset)

## ⚙️ How It Works

1. **User enters symptoms** (comma-separated).
2. The model converts symptoms into a feature vector.
3. A trained classification model predicts the most likely disease.
4. The system fetches:
   - A brief description
   - Suggested diets
   - Medications
   - Precautions to take
   - Recommended workouts

---

## 🛠️ Tech Stack

- **Frontend**: [Streamlit](https://streamlit.io/)
- **Backend**: Python
- **ML Model**: Trained using scikit-learn (SVC, RandomForest, KNN, Multinomial Naive Bayes and Gradient Boosting)
- **Data Handling**: `pandas`, `numpy`
