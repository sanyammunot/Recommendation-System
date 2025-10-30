# CLAT Mentor Recommendation System

**Live Demo**: [clat-mentor-finder.streamlit.app](https://clat-mentor-finder.streamlit.app/)

---

## 📃 Overview

This project is a **Personalized Mentor Recommendation System** designed for CLAT aspirants to connect with suitable mentors based on their preparation preferences and learning styles. The system uses a content-based recommendation approach with clustering and similarity techniques to suggest the top 5 mentors for each aspirant.

---

## 📊 Features

- Processes user input including:
  - Preferred subjects
  - Target colleges
  - Current preparation level
  - Learning style
  - Weak subjects
- Compares against mentor database using:
  - Content-based filtering
  - Cosine similarity
  - KMeans clustering
- Outputs:
  - Top 5 mentor recommendations
  - Explanations for recommendations
- Streamlit UI for interactive user input and results display

---

## 🔧 Tech Stack

- **Python 3.12**
- **Streamlit** for frontend UI
- **Pandas, NumPy** for data processing
- **Scikit-learn** for clustering and similarity calculations
- **Joblib** for model persistence

---

## 🔢 Installation & Setup

```bash
# Clone the repo
https://github.com/Naja24/my_recommendation_system.git
cd my_recommendation_system

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run src/main.py
```

---

## 📚 File Structure

```
my_recommendation_system/
├── data/                       # Contains mock/real anonymized mentor and aspirant data
├── notebooks/   
    └── EDA.py                  # [TO BE IMPLEMENTED] EDA for understanding user-mentor patterns
├── src/
│   ├── main.py                 # Streamlit application
│   ├── recommendation.py       # Core recommendation logic
│   ├── data_models.py          # Creates and Loads Synthesized data
│   ├── model_manager.py
│   ├── evaluation.py           # Evaluation metrics like precision, recall, NDCG
├── tests/
│   ├── test_recommendation.py 
├── requirements.txt            # Dependencies
├── README.md                   # Project documentation
└── .streamlit/
    └── config.toml             # Deployment configurations
```

---

## 📈 Summary of Approach

### 🔍 Data Preprocessing:

- Cleaned and structured aspirant and mentor data
- Encoded categorical fields (e.g., subjects, colleges)
- Normalized numerical fields (e.g., CLAT score, ratings)

### 📊 Clustering & Similarity:

- Mentors were grouped into clusters using KMeans to find similarity zones
- Aspirant profile compared against each cluster and within-cluster mentors using **Cosine Similarity**
- Final recommendation based on combination of subject match, style, weakness coverage, and similarity score

### 🔹 Explanations:

- Provides reasons for each recommendation such as:
  - Subject overlap
  - Style match
  - Experience
  - Ratings and cost

### 🎁 Bonus Evaluation:

- Used metrics such as Average Precision, Recall, and NDCG\@k to validate recommendation quality

---

## 🚀 Future Enhancements & Scope

### 🔄 Feedback Loop

- Incorporate **user feedback** post-session to retrain recommendation model and adapt to user preferences.

### 💡 ML Upgrades

- Integrate **collaborative filtering** using user ratings
- Hybrid system combining content + collaborative techniques
- Train supervised models (e.g., Random Forests) for personalized scoring

### 💡 Improved UI/UX

- Add mentor profile previews with ratings, reviews
- Rating interface after sessions
- Visual analysis and graphs

### 🔍 EDA.py [To Be Implemented]

- This module will:
  - Analyze mentor subject expertise distributions
  - Cluster visualization using t-SNE/PCA
  - Demand vs availability heatmaps (e.g., mentor availability for popular colleges)

---

## 🎯 Evaluation Results (k=5)

```
Precision:  0.0694
Recall:     0.1156
NDCG@k:     0.1388
```

These are baseline results using mock data. Feedback integration and real-world tuning can significantly improve this.

📌 Performance Optimization

Current metrics reflect the baseline on synthetic data. Future improvements include better profile encoding, fine-tuned similarity measures, hybrid filtering models, and real-time user feedback to iteratively improve accuracy and relevance.

---

## 🚜 Contributing

Pull requests and ideas are welcome! Fork the repo and open an issue for suggestions or improvements.

---

## 📖 License

This project is licensed under the MIT License.

