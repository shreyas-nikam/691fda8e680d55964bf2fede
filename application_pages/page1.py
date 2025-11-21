"""import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import plotly.express as px
import plotly.graph_objects as go

def run_page1():
    st.title("Page 1: Setup and Baseline")
    st.markdown("## Notebook Overview")
    st.markdown("""
    This lab aims to implement a scenario-based robustness test for a given machine learning model. Upon completion, users will be able to:

    *   Accept a pre-trained machine learning model and a test dataset as input.
    *   Implement a function to apply user-defined stress transformations to specified input columns.
    *   Define and execute multiple stress scenarios, comparing their prediction distributions to a baseline.
    *   Calculate and display quantitative metrics, such as the mean shift in predictions, for each scenario.
    *   Interpret the results of robustness tests and understand their implications for model design, monitoring, and operational constraints, particularly in contexts like finance.
    """)

    st.markdown("## 3. Setup and Library Imports")
    st.markdown("""
    We begin by importing all the necessary Python libraries for data manipulation, machine learning model operations, and visualization. `pandas` will handle our tabular data, `numpy` for numerical operations, `sklearn` for dataset loading, model training, and data splitting, and `plotly` for creating insightful visualizations of prediction distributions and shifts.
    """)

    st.markdown("## 4. Dataset Loading and Preparation")
    st.markdown("""
    For this lab, we will use the California Housing dataset, which is a tabular dataset ideal for regression tasks. We will prepare it by splitting it into training and testing sets, and identify key numerical columns that will serve as 'volatility columns' for stress testing.
    """)

    @st.cache_resource
    def load_and_prepare_data():
        housing = fetch_california_housing()
        X = pd.DataFrame(housing.data, columns=housing.feature_names)
        y = pd.Series(housing.target)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        vol_cols = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup']
        return X_train, X_test, y_train, y_test, vol_cols

    # Load data into session_state if not already present
    if 'X_test' not in st.session_state:
        st.session_state.X_train, st.session_state.X_test, st.session_state.y_train, \
        st.session_state.y_test, st.session_state.vol_cols = load_and_prepare_data()

    X_test = st.session_state.X_test
    y_test = st.session_state.y_test
    vol_cols = st.session_state.vol_cols

    st.write("First 5 rows of X_test:")
    st.dataframe(X_test.head())
    st.write("\nFirst 5 rows of y_test:")
    st.dataframe(y_test.head())
    st.write("\nVolatility Columns:")
    st.write(vol_cols)

    st.markdown("""
    The California Housing dataset provides several numerical features, which makes it suitable for applying "volatility" shocks. We've selected common features such as Median Income, House Age, and Population as `vol_cols` to simulate how changes in these economic or demographic factors might impact housing price predictions.
    """)

    st.markdown("## 5. Pre-trained Model Initialization and Training")
    st.markdown("""
    A pre-trained model is required to generate predictions. For demonstration purposes, we will train a simple `LinearRegression` model from `sklearn` on our training data.
    """)

    @st.cache_resource
    def train_model(X_train, y_train):
        model = LinearRegression()
        model.fit(X_train, y_train)
        return model

    if 'model' not in st.session_state:
        st.session_state.model = train_model(st.session_state.X_train, st.session_state.y_train)
    model = st.session_state.model

    st.markdown("""
    We have successfully trained a linear regression model. This model will now be used to generate baseline predictions and predictions under various stress scenarios to evaluate its robustness.
    """)

    st.markdown("## 6. Baseline Predictions")
    st.markdown("""
    Before applying any stress, we generate predictions on the original `X_test` dataset. These are our baseline predictions, representing the model's normal behavior, against which all stressed scenarios will be compared.
    """)

    if 'y_hat_base' not in st.session_state:
        st.session_state.y_hat_base = model.predict(X_test)
    y_hat_base = st.session_state.y_hat_base

    st.write("Descriptive statistics of baseline predictions:")
    st.dataframe(pd.Series(y_hat_base).describe())

    st.markdown("""
    The descriptive statistics for `y_hat_base` provide a summary of our model's predictions under normal conditions. This will serve as the crucial reference point for understanding the impact of stress.
    """)

    st.markdown("## 7. Visualize Baseline Predictions")
    st.markdown("""
    Visualizing the distribution of baseline predictions helps us understand the model's typical output range and pattern before any external stressors are applied.
    """)

    # Plotly conversion for KDE plot
    fig = px.histogram(pd.DataFrame({'Baseline Predictions': y_hat_base}),
                       x='Baseline Predictions',
                       nbins=50,
                       histnorm='density',
                       title="Distribution of Baseline Predictions",
                       labels={'Baseline Predictions': "Predicted Value", 'density': "Density"})
    fig.update_traces(marker_color='blue', opacity=0.6)
    st.plotly_chart(fig)

    st.markdown("""
    This plot shows the probability density of our model's predictions on the unstressed test data. Its shape (e.g., unimodal, bimodal, skewed) provides initial insights into the model's output characteristics.
    """)
"""))