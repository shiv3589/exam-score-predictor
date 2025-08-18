# 📚 Exam Score Predictor - Linear Regression Web App

A **Streamlit web application** that demonstrates linear regression for exam score prediction, designed specifically for **12th-grade students**. This educational tool makes machine learning concepts accessible and interactive!

## 🎯 What This App Does

- **Predicts exam scores** based on hours studied using linear regression
- **Interactive visualizations** showing the relationship between study time and scores
- **Educational explanations** of how machine learning works
- **Real-time predictions** with user input
- **Model performance metrics** explained in simple terms

## 🚀 Quick Start

### Option 1: Run Locally (Recommended for Development)

1. **Install Python** (3.7 or higher)

2. **Install required packages:**
   ```bash
   pip install streamlit pandas matplotlib scikit-learn plotly numpy
   ```

3. **Download the code:**
   - Download `app.py` and the `.streamlit/config.toml` file
   - Place them in the same folder

4. **Run the application:**
   ```bash
   streamlit run app.py --server.port 5000
   ```

5. **Open your browser** and go to `http://localhost:5000`

### Option 2: Deploy to Streamlit Cloud (Best for Sharing)

1. **Create a GitHub account** (if you don't have one)

2. **Create a new repository** and upload these files:
   - `app.py`
   - `.streamlit/config.toml`
   - `README.md`

3. **Visit [share.streamlit.io](https://share.streamlit.io)**

4. **Connect your GitHub account** and select your repository

5. **Deploy!** Your app will be live and shareable in minutes

## 📖 Features

### 🏠 Home & Prediction
- Interactive slider to input study hours
- Real-time score predictions
- Encouraging feedback based on predicted scores
- Live visualization of your prediction

### 📊 Data & Visualization
- View the training dataset
- Interactive charts with hover information
- Statistical summary of the data

### 🤖 Model Details
- Model accuracy (R² score)
- Mathematical coefficients explained
- Prediction accuracy on test data
- Error metrics in simple terms

### 📚 How It Works
- Step-by-step explanation of linear regression
- Mathematical formula breakdown
- Real-world applications
- Why machine learning works

### 🚀 Deployment Guide
- Multiple deployment options
- Step-by-step instructions
- Tips for sharing with classmates

## 🎓 Educational Value

This app is perfect for **12th-grade students** because it:

- **Demonstrates real math concepts** (linear equations) in action
- **Shows practical applications** of statistics and data science
- **Encourages experimentation** with different input values
- **Explains complex concepts** in simple, relatable terms
- **Provides hands-on experience** with machine learning

## 🛠️ Technical Details

- **Framework:** Streamlit
- **Machine Learning:** Scikit-learn (Linear Regression)
- **Visualization:** Matplotlib & Plotly
- **Data Handling:** Pandas & NumPy
- **Deployment:** Streamlit Cloud compatible

## 📊 The Dataset

The app uses a simple dataset showing the relationship between study hours and exam scores:

| Hours Studied | Exam Score |
|---------------|------------|
| 1             | 20%        |
| 2             | 40%        |
| 3             | 60%        |
| ...           | ...        |
| 10            | 100%       |

## 🤝 Perfect for Classroom Use

### Teachers can use this to:
- Demonstrate linear regression concepts
- Show real-world applications of math
- Encourage data-driven thinking
- Introduce machine learning basics

### Students can:
- Experiment with different study hours
- See immediate visual feedback
- Understand the math behind predictions
- Share their deployed apps with friends

## 🚀 Deployment Options Summary

| Method | Difficulty | Cost | Best For |
|--------|------------|------|----------|
| **Local** | Easy | Free | Development & Testing |
| **Streamlit Cloud** | Very Easy | Free | Sharing with Classmates |
| **Heroku** | Advanced | Free Tier | Permanent Hosting |

## 🆘 Troubleshooting

### Common Issues:

1. **"Module not found" error:**
   ```bash
   pip install streamlit pandas matplotlib scikit-learn plotly numpy
   