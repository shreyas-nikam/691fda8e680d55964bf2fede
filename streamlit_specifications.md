
# Streamlit Application Requirements Specification

## 1. Application Overview

This Streamlit application will provide an interactive environment for users to perform scenario-based robustness tests on machine learning models. It will guide users through the process of defining stress scenarios, applying transformations to input data, generating stressed predictions, and quantitatively and visually assessing the impact on model stability.

**Learning Goals:**
Upon completion of using this application, users will be able to:
*   Accept a pre-trained machine learning model and a test dataset as input.
*   Implement a function to apply user-defined stress transformations to specified input columns.
*   Define and execute multiple stress scenarios, comparing their prediction distributions to a baseline.
*   Calculate and display quantitative metrics, such as the mean shift in predictions, for each scenario.
*   Interpret the results of robustness tests and understand their implications for model design, monitoring, and operational constraints, particularly in contexts like finance.

**Robustness Concept:**
Given a model $\hat{y} = f_\theta(x)$, we wish to evaluate its stability under stress transformations $T_s$ applied to inputs (e.g., volatility shocks). For each scenario $s$:
$$x^{(s)} = T_s(x), \\quad \\hat{y}^{(s)} = f_\theta(x^{(s)}).$$
We compare distributions of $\hat{y}^{(s)}$ to baseline predictions.

## 2. User Interface Requirements

### Layout and Navigation Structure
The application will follow a single-page layout with a clear progression of steps.
*   **Main Content Area:** Will display markdown explanations, data previews, and all visualizations.
*   **Sidebar:** Will house configuration inputs for dataset loading, model selection, defining volatility columns, and managing stress scenarios.

### Input Widgets and Controls
The sidebar will contain interactive widgets for user input:

*   **Data and Model Setup:**
    *   **Dataset Selection:** A radio button or dropdown to select between "Load California Housing Dataset" (default) or "Upload Custom Dataset (CSV/Parquet)".
    *   **File Uploader (if custom dataset selected):** `st.file_uploader` for `X_test` and `y_test`.
    *   **Volatility Columns Selection:** `st.multiselect` to choose columns from `X_test` to be designated as 'volatility columns' for stress testing. This will be pre-populated with default `vol_cols` for California Housing.

*   **Stress Scenario Definition:**
    *   **Scenario Type:** `st.selectbox` with options like "Volatility Multiplier" or "Add Gaussian Noise".
    *   **Factor/Std Dev Input:** `st.number_input` for `factor` (for volatility, e.g., default `2.0` or `0.5`) or `noise_std_dev` (for noise, e.g., default `0.5`), depending on the selected scenario type.
    *   **Columns to Stress (for custom scenario):** `st.multiselect` to choose specific columns for the custom stress.
    *   **Scenario Name:** `st.text_input` for a unique name for the new custom scenario.
    *   **Add Scenario Button:** `st.button` to add the defined custom scenario to the list of active scenarios.

*   **Execution Control:**
    *   **Run Robustness Test Button:** `st.button` to trigger the execution of all defined and active stress scenarios, predictions, and analysis.

### Visualization Components
The main content area will display:

*   **Data Tables:** `st.dataframe` for `X_test.head()`, `y_test.head()`, and descriptive statistics of baseline predictions.
*   **Plots:**
    *   **Baseline Prediction Distribution:** `st.pyplot` displaying a Kernel Density Estimate (KDE) plot of `y_hat_base`.
    *   **Scenario-Specific Prediction Distributions:** For each active stress scenario, `st.pyplot` showing an overlaid KDE plot comparing `y_hat_base` and `y_hat_stress`.
    *   **Mean Shift Bar Chart:** `st.pyplot` displaying a bar chart of mean shifts across all scenarios, visually comparing their quantitative impact.
*   **Quantitative Metrics:** `st.write` or `st.metric` to display calculated mean shifts for each scenario.

### Interactive Elements and Feedback Mechanisms
*   **Dynamic UI Updates:** Input widgets and visualizations will update dynamically based on user selections (e.g., `vol_cols` options changing based on `X_test`).
*   **Loading Indicators:** `st.spinner` or `st.status` will be used during computationally intensive operations like model training, prediction generation, and scenario execution.
*   **Information Messages:** `st.info` or `st.success` will provide feedback on successful operations (e.g., "Dataset loaded," "Model trained," "Scenario added").
*   **Error Handling:** `st.error` will display informative messages in case of invalid inputs or execution failures.

## 3. Additional Requirements

*   **Annotation and Tooltip Specifications:**
    *   **Plot Annotations:** Each overlaid KDE plot will include an annotation displaying the `Mean Shift` value (e.g., `Mean Shift: {shift:.2f}`).
    *   **Tooltips/Help Text:** `st.help` or clear `st.markdown` explanations will be provided for input parameters (e.g., "Enter a multiplicative factor for the volatility shock. Factor > 1.0 increases values, Factor < 1.0 decreases values.").
    *   **Explanatory Sections:** Markdown cells from the notebook, particularly the "Robustness Concept" and "Key Points" at the end, will be rendered using `st.markdown` to provide context and interpretation.

*   **Save the states of the fields properly so that changes are not lost:**
    *   `st.session_state` will be extensively used to maintain the application's state across reruns. This includes:
        *   The loaded `X_train`, `X_test`, `y_train`, `y_test` DataFrames.
        *   The trained `model` object.
        *   The baseline predictions `y_hat_base`.
        *   The dictionary of active `scenarios`, including user-defined ones and their parameters.
        *   The `stressed_predictions` and `mean_shifts` results.
        *   User selections for `vol_cols` and other parameters.

## 4. Notebook Content and Code Requirements

All markdown content from the Jupyter Notebook will be included in the Streamlit application using `st.markdown()`. Code stubs are extracted below to illustrate their integration into the Streamlit app's logical flow.

### Application Overview and Setup
```python
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns

# Set global options for reproducibility
np.random.seed(42)

st.title("# Scenario-Based Model Robustness Test")
st.markdown("## Notebook Overview")
st.markdown("""
This lab aims to implement a scenario-based robustness test for a given machine learning model. Upon completion, users will be able to:

*   Accept a pre-trained machine learning model and a test dataset as input.
*   Implement a function to apply user-defined stress transformations to specified input columns.
*   Define and execute multiple stress scenarios, comparing their prediction distributions to a baseline.
*   Calculate and display quantitative metrics, such as the mean shift in predictions, for each scenario.
*   Interpret the results of robustness tests and understand their implications for model design, monitoring, and operational constraints, particularly in contexts like finance.
""")
# Robustness Concept markdown (from OCR image and user prompt)
st.markdown("## Robustness Concept")
st.markdown("""
Given a model $\\hat{y} = f_\\theta(x)$, we wish to evaluate its stability under stress transformations $T_s$ applied to inputs (e.g., volatility shocks). For each scenario $s$:
$$x^{(s)} = T_s(x), \\quad \\hat{y}^{(s)} = f_\\theta(x^{(s)}).$$
We compare distributions of $\\hat{y}^{(s)}$ to baseline.
""")

st.markdown("## 3. Setup and Library Imports")
st.markdown("""
We begin by importing all the necessary Python libraries for data manipulation, machine learning model operations, and visualization. `pandas` will handle our tabular data, `numpy` for numerical operations, `sklearn` for dataset loading, model training, and data splitting, and `matplotlib` along with `seaborn` for creating insightful visualizations of prediction distributions and shifts.
""")
```

### Dataset Loading and Preparation
```python
@st.cache_resource
def load_and_prepare_data():
    housing = fetch_california_housing()
    X = pd.DataFrame(housing.data, columns=housing.feature_names)
    y = pd.Series(housing.target)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    vol_cols = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup']
    return X_train, X_test, y_train, y_test, vol_cols

st.markdown("## 4. Dataset Loading and Preparation")
st.markdown("""
For this lab, we will use the California Housing dataset, which is a tabular dataset ideal for regression tasks. We will prepare it by splitting it into training and testing sets, and identify key numerical columns that will serve as 'volatility columns' for stress testing.
""")

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
```

### Pre-trained Model Initialization and Training
```python
@st.cache_resource
def train_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

st.markdown("## 5. Pre-trained Model Initialization and Training")
st.markdown("""
A pre-trained model is required to generate predictions. For demonstration purposes, we will train a simple `LinearRegression` model from `sklearn` on our training data.
""")

if 'model' not in st.session_state:
    st.session_state.model = train_model(st.session_state.X_train, st.session_state.y_train)
model = st.session_state.model

st.markdown("""
We have successfully trained a linear regression model. This model will now be used to generate baseline predictions and predictions under various stress scenarios to evaluate its robustness.
""")
```

### Baseline Predictions and Visualization
```python
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

fig, ax = plt.subplots(figsize=(10, 6))
sns.kdeplot(y_hat_base, fill=True, color='blue', alpha=0.6, ax=ax)
ax.set_title("Distribution of Baseline Predictions")
ax.set_xlabel("Predicted Value")
ax.set_ylabel("Density")
st.pyplot(fig)

st.markdown("""
This plot shows the probability density of our model's predictions on the unstressed test data. Its shape (e.g., unimodal, bimodal, skewed) provides initial insights into the model's output characteristics.
""")
```

### Implementing Stress Transformation Functions
```python
st.markdown("## 8. Implementing the Stress Transformation Function (`stress_scenario_volatility`)")
st.markdown("""
The core of our robustness test is the ability to transform input features to simulate stress. We'll implement a versatile function, `stress_scenario_volatility`, that allows us to multiply specified columns by a chosen factor. This simulates 'volatility shocks' such as sudden economic upturns or downturns.
""")

def stress_scenario_volatility(X_data, factor, vol_cols):
    stressed_df = X_data.copy(deep=True)
    if vol_cols is None:
        target_cols = stressed_df.select_dtypes(include=np.number).columns
    else:
        target_cols = vol_cols
    if len(target_cols) > 0:
        stressed_df[target_cols] = stressed_df[target_cols] * factor
    return stressed_df

st.code("""
def stress_scenario_volatility(X_data, factor, vol_cols):
    \"\"\"
    Applies a volatility shock to specified columns of a Pandas DataFrame.
    Arguments:
    X_data: Pandas DataFrame, the input features to be stressed.
    factor: float, the multiplicative factor.
    vol_cols: list of strings, names of columns to stress. If None, all numerical columns.
    Output:
    Pandas DataFrame, with specified columns modified by the factor.
    \"\"\"
    stressed_df = X_data.copy(deep=True)
    if vol_cols is None:
        target_cols = stressed_df.select_dtypes(include=np.number).columns
    else:
        target_cols = vol_cols
    if len(target_cols) > 0:
        stressed_df[target_cols] = stressed_df[target_cols] * factor
    return stressed_df
""")

st.markdown("""
The `stress_scenario_volatility` function is designed to be flexible. By changing the `factor`, we can simulate both increases (e.g., `factor=2.0` for a 100% increase) and decreases (e.g., `factor=0.5` for a 50% decrease) in the 'volatility columns'. This function will be the building block for defining our stress scenarios.
""")
```

### Defining Stress Scenarios
```python
st.markdown("## 9. Defining Stress Scenarios")
st.markdown("""
Now we define a dictionary of stress scenarios. Each scenario is a lambda function that applies our `stress_scenario_volatility` function (or other custom stress functions) with specific parameters to a copy of the `X_test` data. This allows for easy configuration and expansion of test cases.
""")

# Initialize default scenarios in session state
if 'scenarios' not in st.session_state:
    st.session_state.scenarios = {
        'vol_up': {'func_name': 'stress_scenario_volatility', 'params': {'factor': 2.0, 'vol_cols': vol_cols}},
        'vol_down': {'func_name': 'stress_scenario_volatility', 'params': {'factor': 0.5, 'vol_cols': vol_cols}},
        'single_col_spike': {'func_name': 'stress_scenario_volatility', 'params': {'factor': 1.5, 'vol_cols': ['MedInc']}}
    }

# Sidebar for adding custom scenarios
with st.sidebar.expander("Define Custom Stress Scenario"):
    st.markdown("### Add New Scenario")
    scenario_type = st.selectbox("Select Stress Type", ["Volatility Multiplier", "Add Gaussian Noise"])
    
    new_vol_cols = st.multiselect("Columns to Stress", X_test.columns.tolist(), default=vol_cols)
    new_scenario_name = st.text_input("New Scenario Name", value=f"{scenario_type.replace(' ', '_').lower()}_{len(st.session_state.scenarios)+1}")
    
    new_scenario_params = {}
    new_scenario_func_name = ''

    if scenario_type == "Volatility Multiplier":
        new_factor = st.number_input("Multiplicative Factor", min_value=0.01, max_value=10.0, value=1.5, step=0.1)
        new_scenario_params = {'factor': new_factor, 'vol_cols': new_vol_cols}
        new_scenario_func_name = 'stress_scenario_volatility'
    elif scenario_type == "Add Gaussian Noise":
        new_noise_std_dev = st.number_input("Noise Standard Deviation", min_value=0.0, max_value=5.0, value=0.5, step=0.1)
        new_scenario_params = {'noise_std_dev': new_noise_std_dev, 'vol_cols': new_vol_cols}
        new_scenario_func_name = 'stress_scenario_add_noise'

    if st.button("Add Scenario to List"):
        if new_scenario_name in st.session_state.scenarios:
            st.warning(f"Scenario '{new_scenario_name}' already exists. Please choose a different name.")
        elif not new_scenario_name:
            st.warning("Scenario name cannot be empty.")
        else:
            st.session_state.scenarios[new_scenario_name] = {
                'func_name': new_scenario_func_name,
                'params': new_scenario_params
            }
            st.success(f"Scenario '{new_scenario_name}' added!")

st.markdown("We've defined three distinct scenarios: a general 'vol_up' shock across all designated volatility columns, a 'vol_down' shock, and a targeted 'single_col_spike' affecting only the `MedInc` (Median Income) column. These scenarios simulate different types of potential real-world events that could impact the model's inputs.")
```

### Executing Stress Scenarios and Generating Stressed Predictions
```python
st.markdown("## 10. Executing Stress Scenarios and Generating Stressed Predictions")
st.markdown("""
With our scenarios defined, we now execute each one. For every scenario, we apply the stress transformation to a copy of `X_test` and then generate new predictions using our pre-trained model. These stressed predictions will then be compared to our baseline.
""")

# Map function names to actual functions
scenario_functions = {
    'stress_scenario_volatility': stress_scenario_volatility,
    'stress_scenario_add_noise': stress_scenario_add_noise
}

def execute_scenarios(model, X_test, scenarios_dict):
    stressed_predictions = {}
    for name, details in scenarios_dict.items():
        func = scenario_functions.get(details['func_name'])
        if func:
            X_stress = func(X_test.copy(), **details['params'])
            y_hat_stress = model.predict(X_stress)
            stressed_predictions[name] = y_hat_stress
    return stressed_predictions

if st.sidebar.button("Run Robustness Test"):
    with st.spinner("Running scenarios and generating predictions..."):
        st.session_state.stressed_predictions = execute_scenarios(model, X_test, st.session_state.scenarios)
    st.success("Robustness test completed!")

stressed_predictions = st.session_state.get('stressed_predictions', {})

if stressed_predictions:
    st.write("Scenarios processed:", list(stressed_predictions.keys()))
    if 'vol_up' in stressed_predictions:
        st.write("\nDescriptive statistics for 'vol_up' stressed predictions:")
        st.dataframe(pd.Series(stressed_predictions['vol_up']).describe())

st.markdown("""
We have successfully applied each defined stress scenario and collected the corresponding model predictions. Observing the statistics for a sample stressed scenario, like 'vol_up', already hints at how the model's output distribution might have shifted compared to the baseline.
""")
```

### Calculating Quantitative Shift Metrics
```python
st.markdown("## 11. Calculating Quantitative Shift Metrics")
st.markdown("""
While visual comparisons are intuitive, quantitative metrics provide a concise summary of the impact. A common metric is the **mean shift** in predictions, which indicates the average change in the model's output due to the stress.
""")

def calculate_mean_shift(baseline_predictions, stressed_predictions):
    mean_baseline = baseline_predictions.mean()
    mean_stressed = stressed_predictions.mean()
    return float(mean_stressed - mean_baseline)

st.code("""
def calculate_mean_shift(baseline_predictions, stressed_predictions):
    \"\"\"
    Calculates the quantitative difference between the mean of stressed predictions
    and the mean of baseline predictions.
    Arguments:
    baseline_predictions: NumPy array or Pandas Series, the model's predictions on
                          the original, unstressed test dataset.
    stressed_predictions: NumPy array or Pandas Series, the model's predictions on
                          the test dataset after applying a specific stress transformation.
    Output:
    float, a scalar value representing the difference between the mean of stressed
    predictions and baseline predictions.
    \"\"\"
    mean_baseline = baseline_predictions.mean()
    mean_stressed = stressed_predictions.mean()
    return float(mean_stressed - mean_baseline)
""")

if 'stressed_predictions' in st.session_state and stressed_predictions and 'mean_shifts' not in st.session_state:
    mean_shifts = {}
    for name, y_hat_stress in stressed_predictions.items():
        shift = calculate_mean_shift(y_hat_base, y_hat_stress)
        mean_shifts[name] = shift
    st.session_state.mean_shifts = mean_shifts

mean_shifts = st.session_state.get('mean_shifts', {})
if mean_shifts:
    st.write("Calculated Mean Shifts:")
    st.json(mean_shifts)

st.markdown("""
The `mean_shifts` dictionary clearly shows how the average prediction changed for each scenario. A positive value indicates an increase in the average prediction, while a negative value indicates a decrease. These values provide the first quantitative measure of our model's robustness.
""")
```

### Visualizing Prediction Distributions for Each Scenario
```python
st.markdown("## 12. Visualizing Prediction Distributions for Each Scenario")
st.markdown("""
To gain a deeper understanding of the impact of each stress scenario, we will visualize the distribution of stressed predictions against the baseline. This allows us to observe not just mean shifts, but also changes in variance, skewness, or the emergence of new modes in the predictions.
""")

if stressed_predictions and mean_shifts:
    for name, y_hat_stress in stressed_predictions.items():
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.kdeplot(y_hat_base, fill=True, color='blue', alpha=0.5, label="Baseline", ax=ax)
        sns.kdeplot(y_hat_stress, fill=True, color='red', alpha=0.5, label=name, ax=ax)
        
        shift = mean_shifts.get(name, 0.0)
        
        ax.set_title(f"Prediction Distribution: Baseline vs. {name}")
        ax.set_xlabel("Predicted Value")
        ax.set_ylabel("Density")
        ax.legend()
        ax.text(0.05, 0.9, f'Mean Shift: {shift:.2f}', transform=ax.transAxes, fontsize=12, 
                 bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
        st.pyplot(fig)

st.markdown("""
These overlaid plots visually confirm the shifts quantified earlier. They also reveal more subtle changes, such as whether the stress scenario caused predictions to become more spread out (increased variance), skewed, or if certain prediction values became more or less common.
""")
```

### Visualizing Quantitative Shifts (Bar Chart)
```python
st.markdown("## 13. Visualizing Quantitative Shifts (Bar Chart)")
st.markdown("""
A bar chart of the mean shifts provides a consolidated view, making it easy to compare the magnitude of impact across all scenarios and quickly identify the most disruptive ones.
""")

if mean_shifts:
    mean_shifts_series = pd.Series(mean_shifts)
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.barplot(x=mean_shifts_series.index, y=mean_shifts_series.values, palette='viridis', ax=ax)
    ax.set_title("Mean Shift in Predictions Across Scenarios")
    ax.set_xlabel("Stress Scenario")
    ax.set_ylabel("Mean Shift")
    plt.xticks(rotation=45, ha='right')
    ax.axhline(0, color='gray', linestyle='--')
    plt.tight_layout()
    st.pyplot(fig)

st.markdown("""
This bar chart offers a clear, at-a-glance summary of how each stress scenario affected the average model prediction. Scenarios with larger absolute mean shifts indicate areas where the model's stability is more significantly compromised.
""")
```

### Customizing Stress Scenarios (Advanced)
```python
st.markdown("## 14. Customizing Stress Scenarios (Advanced)")
st.markdown("""
The flexibility of this framework allows for defining various types of stress beyond simple multiplicative factors. Users can implement custom functions to simulate more complex real-world phenomena, such as adding random noise, applying thresholds, or simulating specific market event impacts.
""")

def stress_scenario_add_noise(X_data, noise_std_dev, vol_cols):
    X_stressed = X_data.copy(deep=True)
    if vol_cols is None:
        target_cols = X_stressed.select_dtypes(include=np.number).columns
    else:
        target_cols = vol_cols
    for col in target_cols:
        noise = np.random.normal(loc=0, scale=noise_std_dev, size=len(X_stressed))
        X_stressed[col] = X_stressed[col] + noise
    return X_stressed

st.code("""
def stress_scenario_add_noise(X_data, noise_std_dev, vol_cols):
    \"\"\"    This function introduces random Gaussian noise to specified columns of a Pandas DataFrame, simulating unpredictable measurement errors or market fluctuations. It creates a deep copy of the input DataFrame to avoid modifying the original data, preserving the original test set for baseline comparisons.
Arguments:
X_data: Pandas DataFrame, the input features to be stressed.
noise_std_dev: float, the standard deviation of the Gaussian noise to add (mean is 0).
vol_cols: list of strings, the names of the columns to add noise to. If None, noise is added to all numerical columns in the DataFrame.
Output:
Pandas DataFrame, the DataFrame with specified columns having added random Gaussian noise.
    \"\"\"
    X_stressed = X_data.copy(deep=True)
    if vol_cols is None:
        target_cols = X_stressed.select_dtypes(include=np.number).columns
    else:
        target_cols = vol_cols
    for col in target_cols:
        noise = np.random.normal(loc=0, scale=noise_std_dev, size=len(X_stressed))
        X_stressed[col] = X_stressed[col] + noise
    return X_stressed
""")

st.markdown("""
The `stress_scenario_add_noise` function introduces random Gaussian noise to specified features, simulating unpredictable measurement errors or market fluctuations. By adding this new scenario, we demonstrate how easily the robustness test framework can be extended to cover a wider range of potential stressors.
""")
```

### Rerunning Analysis with Custom Scenarios
This section is implicitly handled by the "Run Robustness Test" button. When the button is clicked, it re-executes the prediction and analysis steps, incorporating any newly defined custom scenarios stored in `st.session_state.scenarios`.

```python
st.markdown("## 15. Rerunning Analysis with Custom Scenarios")
st.markdown("""
To see the impact of our newly defined custom stress scenario, we will re-execute the prediction and analysis steps, incorporating the 'noise_shock'. This iterative process is crucial for thoroughly evaluating model robustness under diverse conditions.
""")

# The 'Run Robustness Test' button in the sidebar will handle this step
# Display updated mean shifts if available
if mean_shifts:
    st.write("Updated Calculated Mean Shifts (including custom scenarios):")
    st.json(mean_shifts)

st.markdown("""
The updated `mean_shifts` now include the impact of any newly added custom scenarios. This iterative approach allows us to dynamically assess new stress scenarios and their effects, providing a comprehensive view of the model's behavior.
""")
```
