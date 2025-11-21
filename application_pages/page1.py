
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

@st.cache_resource
def load_and_prepare_data():
    housing = fetch_california_housing()
    X = pd.DataFrame(housing.data, columns=housing.feature_names)
    y = pd.Series(housing.target)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    vol_cols = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup']
    return X_train, X_test, y_train, y_test, vol_cols

@st.cache_resource
def train_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def run_page1():
    st.title("Application Overview & Setup")
    st.markdown("""
    ## Notebook Overview
    This lab aims to implement a scenario-based robustness test for a given machine learning model. Upon completion, users will be able to:

    *   Accept a pre-trained machine learning model and a test dataset as input.
    *   Implement a function to apply user-defined stress transformations to specified input columns.
    *   Define and execute multiple stress scenarios, comparing their prediction distributions to a baseline.
    *   Calculate and display quantitative metrics, such as the mean shift in predictions, for each scenario.
    *   Interpret the results of robustness tests and understand their implications for model design, monitoring, and operational constraints, particularly in contexts like finance.
    """)
    st.markdown("""
    ## Robustness Concept
    Given a model $\hat{y} = f_\theta(x)$, we wish to evaluate its stability under stress transformations $T_s$ applied to inputs (e.g., volatility shocks). For each scenario $s$:
    $$x^{(s)} = T_s(x), \quad \hat{y}^{(s)} = f_\theta(x^{(s)}).$$
    We compare distributions of $\hat{y}^{(s)}$ to baseline.
    """)

    st.markdown("""
    ## 3. Setup and Library Imports
    We begin by importing all the necessary Python libraries for data manipulation, machine learning model operations, and visualization. `pandas` will handle our tabular data, `numpy` for numerical operations, `sklearn` for dataset loading, model training, and data splitting, and `plotly` for creating insightful visualizations of prediction distributions and shifts.
    """)

    st.markdown("""
    ## 4. Dataset Loading and Preparation
    For this lab, we will use the California Housing dataset, which is a tabular dataset ideal for regression tasks. We will prepare it by splitting it into training and testing sets, and identify key numerical columns that will serve as 'volatility columns' for stress testing.
    """)

    if 'X_test' not in st.session_state:
        with st.spinner("Loading and preparing data..."):
            st.session_state.X_train, st.session_state.X_test, st.session_state.y_train, \
            st.session_state.y_test, st.session_state.vol_cols = load_and_prepare_data()
        st.success("California Housing Dataset loaded!")

    X_test = st.session_state.X_test
    # y_test = st.session_state.y_test # Not strictly needed for display, but good for consistency
    vol_cols = st.session_state.vol_cols

    st.write("First 5 rows of X_test:")
    st.dataframe(X_test.head())
    st.write("First 5 rows of y_test:")
    st.dataframe(st.session_state.y_test.head()) # Use st.session_state.y_test directly
    st.write("Volatility Columns:")
    st.write(vol_cols)

    st.markdown("""
    The California Housing dataset provides several numerical features, which makes it suitable for applying "volatility" shocks. We've selected common features such as Median Income, House Age, and Population as `vol_cols` to simulate how changes in these economic or demographic factors might impact housing price predictions.
    """)

    st.markdown("""
    ## 5. Pre-trained Model Initialization and Training
    A pre-trained model is required to generate predictions. For demonstration purposes, we will train a simple `LinearRegression` model from `sklearn` on our training data.
    """)

    if 'model' not in st.session_state:
        with st.spinner("Training model..."):
            st.session_state.model = train_model(st.session_state.X_train, st.session_state.y_train)
        st.success("Model trained successfully!")
    # model = st.session_state.model # Not strictly needed for display here

    st.markdown("""
    We have successfully trained a linear regression model. This model will now be used to generate baseline predictions and predictions under various stress scenarios to evaluate its robustness.
    """)
