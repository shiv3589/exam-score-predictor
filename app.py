import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Exam Score Predictor",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main title and introduction
st.title("📚 Exam Score Predictor")
st.markdown("### Learn Linear Regression with Real Examples!")

st.markdown("""
Welcome to the **Exam Score Predictor**! This application demonstrates how **Linear Regression** 
works by predicting exam scores based on hours studied. Perfect for understanding machine learning concepts!
""")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choose a section:", [
    "🏠 Home & Prediction", 
    "📊 Data & Visualization", 
    "🤖 Model Details", 
    "📖 How It Works",
    "🚀 Deployment Guide"
])

# Create and prepare the data (same as in the notebook)
@st.cache_data
def load_data():
    """Load and prepare the sample data"""
    data = {
        "Hours_Studied": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "Exam_Score": [20, 40, 60, 65, 70, 80, 85, 90, 95, 100]
    }
    return pd.DataFrame(data)

@st.cache_data
def train_model(df):
    """Train the linear regression model"""
    X = df[["Hours_Studied"]]
    y = df["Exam_Score"]
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    score = model.score(X_test, y_test)
    
    return model, X_train, X_test, y_train, y_test, y_pred, score

# Load data and train model
df = load_data()
model, X_train, X_test, y_train, y_test, y_pred, score = train_model(df)

# Page content based on selection
if page == "🏠 Home & Prediction":
    st.header("🎯 Predict Your Exam Score!")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Enter Your Study Hours")
        hours_studied = st.slider(
            "How many hours will you study?", 
            min_value=0.5, 
            max_value=15.0, 
            value=5.0, 
            step=0.5,
            help="Move the slider to see how study hours affect your predicted score!"
        )
        
        # Make prediction
        predicted_score = model.predict([[hours_studied]])[0]
        
        # Display prediction with nice formatting
        st.metric(
            label="Predicted Exam Score", 
            value=f"{predicted_score:.1f}%",
            help="This is what our model predicts based on historical data!"
        )
        
        # Add some encouraging messages
        if predicted_score >= 90:
            st.success("🌟 Excellent! You're on track for a great score!")
        elif predicted_score >= 70:
            st.info("👍 Good work! A solid score is within reach!")
        elif predicted_score >= 50:
            st.warning("📈 You're getting there! Consider studying a bit more.")
        else:
            st.error("⚠️ More study time needed for a passing grade!")
    
    with col2:
        st.subheader("Quick Visualization")
        
        # Create a simple plot showing the prediction
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Plot original data
        ax.scatter(df["Hours_Studied"], df["Exam_Score"], color='blue', alpha=0.7, label='Training Data')
        
        # Plot regression line
        x_line = np.linspace(0, 12, 100)
        y_line = model.predict(x_line.reshape(-1, 1))
        ax.plot(x_line, y_line, color='red', linewidth=2, label='Regression Line')
        
        # Highlight the prediction
        ax.scatter([hours_studied], [predicted_score], color='green', s=100, 
                  label=f'Your Prediction ({hours_studied}h → {predicted_score:.1f}%)', zorder=5)
        
        ax.set_xlabel('Hours Studied')
        ax.set_ylabel('Exam Score (%)')
        ax.set_title('Exam Score vs Study Hours')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 12)
        ax.set_ylim(0, 110)
        
        st.pyplot(fig)

elif page == "📊 Data & Visualization":
    st.header("📊 Our Training Data")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Sample Data")
        st.markdown("This is the data we used to train our model:")
        st.dataframe(df, use_container_width=True)
        
        st.subheader("Data Statistics")
        st.write(df.describe())
    
    with col2:
        st.subheader("Interactive Visualization")
        
        # Create an interactive plotly chart
        fig = go.Figure()
        
        # Add scatter plot
        fig.add_trace(go.Scatter(
            x=df["Hours_Studied"],
            y=df["Exam_Score"],
            mode='markers',
            name='Actual Data Points',
            marker=dict(size=10, color='blue'),
            hovertemplate='Hours: %{x}<br>Score: %{y}%<extra></extra>'
        ))
        
        # Add regression line
        x_line = np.linspace(0, 12, 100)
        y_line = model.predict(x_line.reshape(-1, 1))
        fig.add_trace(go.Scatter(
            x=x_line,
            y=y_line,
            mode='lines',
            name='Regression Line',
            line=dict(color='red', width=3)
        ))
        
        fig.update_layout(
            title="Interactive: Hours Studied vs Exam Score",
            xaxis_title="Hours Studied",
            yaxis_title="Exam Score (%)",
            hovermode='closest'
        )
        
        st.plotly_chart(fig, use_container_width=True)

elif page == "🤖 Model Details":
    st.header("🤖 Model Performance & Details")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Model Accuracy")
        st.metric("R² Score (Accuracy)", f"{score * 100:.2f}%")
        
        if score > 0.95:
            st.success("🎯 Excellent accuracy! The model fits the data very well.")
        elif score > 0.8:
            st.info("👍 Good accuracy! The model is reliable for predictions.")
        else:
            st.warning("⚠️ Moderate accuracy. Predictions may vary.")
        
        st.subheader("Model Coefficients")
        st.write(f"**Slope (Coefficient):** {model.coef_[0]:.2f}")
        st.write(f"**Intercept:** {model.intercept_:.2f}")
        
        st.markdown("""
        **What this means:**
        - For every additional hour studied, the exam score increases by approximately **{:.1f} points**
        - If you study 0 hours, the model predicts a score of **{:.1f}%**
        """.format(model.coef_[0], model.intercept_))
    
    with col2:
        st.subheader("Training vs Testing Performance")
        
        # Show prediction comparison
        comparison_df = pd.DataFrame({
            "Actual Score": y_test.values,
            "Predicted Score": y_pred,
            "Difference": y_test.values - y_pred
        })
        
        st.write("**Test Set Predictions:**")
        st.dataframe(comparison_df, use_container_width=True)
        
        # Calculate and show error metrics
        mae = np.mean(np.abs(comparison_df["Difference"]))
        st.metric("Mean Absolute Error", f"{mae:.2f} points")

elif page == "📖 How It Works":
    st.header("📖 Understanding Linear Regression")
    
    st.markdown("""
    ### What is Linear Regression? 🤔
    
    Linear regression is a simple but powerful **machine learning algorithm** that finds the best line 
    through a set of data points. Think of it as drawing the "line of best fit" through scattered points!
    
    ### The Mathematical Formula 📐
    
    Our model uses this equation:
    """)
    
    st.latex(r"Exam\ Score = {:.2f} \times Hours\ Studied + {:.2f}".format(model.coef_[0], model.intercept_))
    
    st.markdown("""
    ### Step-by-Step Process 🔄
    
    1. **Collect Data**: We gathered information about study hours and corresponding exam scores
    2. **Train the Model**: The algorithm finds the best line that fits through our data points
    3. **Make Predictions**: We use the line equation to predict scores for new study hours
    4. **Evaluate Performance**: We check how accurate our predictions are
    
    ### Why Does This Work? 🧠
    
    - **Pattern Recognition**: The model identifies that more study hours generally lead to higher scores
    - **Mathematical Optimization**: It finds the line that minimizes prediction errors
    - **Generalization**: Once trained, it can predict scores for any number of study hours
    
    ### Real-World Applications 🌍
    
    Linear regression is used in many fields:
    - **Economics**: Predicting sales based on advertising spend
    - **Medicine**: Relating dosage to treatment effectiveness  
    - **Sports**: Predicting performance based on training hours
    - **Education**: Understanding factors that affect academic performance
    """)

elif page == "🚀 Deployment Guide":
    st.header("🚀 How to Deploy This App")
    
    st.markdown("""
    ### Option 1: Streamlit Cloud (Easiest) ☁️
    
    **Perfect for sharing with classmates!**
    
    1. **Create a GitHub account** (if you don't have one)
    2. **Upload your code** to a new GitHub repository
    3. **Visit [share.streamlit.io](https://share.streamlit.io)**
    4. **Connect your GitHub** and select your repository
    5. **Deploy!** Your app will be live in minutes
    
    ✅ **Pros**: Free, easy, automatic updates  
    ❌ **Cons**: Requires GitHub account
    
    ---
    
    ### Option 2: Run Locally 💻
    
    **For testing and development:**
    
    ```bash
    # Install required packages
    pip install streamlit pandas matplotlib scikit-learn plotly numpy
    
    # Run the application
    streamlit run app.py --server.port 5000
    ```
    
    ✅ **Pros**: Complete control, no internet needed  
    ❌ **Cons**: Only accessible on your computer
    
    ---
    
    ### Option 3: Replit (For School Projects) 🏫
    
    **Perfect for coding assignments:**
    
    1. **Go to [replit.com](https://replit.com)**
    2. **Create a new Python project**
    3. **Upload your `app.py` file**
    4. **Install packages** by running:
       ```bash
       pip install streamlit pandas matplotlib scikit-learn plotly numpy
       ```
    5. **Run the app** with:
       ```bash
       streamlit run app.py --server.port 5000
       ```
    
    ✅ **Pros**: Works on any computer, great for school  
    ❌ **Cons**: Requires account, limited free usage
    
    ---
    
    ### 🎯 Which Option Should You Choose?
    
    - **Just learning?** → Start with **Local** (Option 2)
    - **Want to share?** → Use **Streamlit Cloud** (Option 1)  
    - **School project?** → Try **Replit** (Option 3)
    - **Need help?** → Ask your teacher which platform they prefer!
    
    ### 📋 Before You Deploy - Checklist
    
    Make sure you have these files:
    - ✅ `app.py` (your main application)
    - ✅ `.streamlit/config.toml` (configuration file)
    - ✅ `README.md` (instructions for others)
    
    ### 🚨 Common Deployment Issues
    
    **Problem**: "Module not found error"  
    **Solution**: Install missing packages with pip
    
    **Problem**: "Port already in use"  
    **Solution**: Change the port number or restart your computer
    
    **Problem**: "App won't load"  
    **Solution**: Check that all files are in the same folder
    
    ### 🎉 Congratulations!
    
    Once deployed, you can:
    - Share the link with classmates
    - Use it for presentations
    - Modify the code to add new features
    - Learn more about machine learning!
    
    **Happy coding and learning!** 🚀
    """)
    
    # Add footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center'>
        <p><b>🎓 Built for Learning | 📊 Powered by Linear Regression | 🚀 Made with Streamlit</b></p>
        <p><i>Perfect for 12th-grade students learning about machine learning!</i></p>
    </div>
    """, unsafe_allow_html=True)