"""import streamlit as st
import pandas as pd
import numpy as np

def run_page2():
    st.title("Page 2: Define Scenarios")
    
    # Ensure X_test and vol_cols are in session_state, as they are needed here.
    # This assumes page1 has been run or the app has initialized session_state
    if 'X_test' not in st.session_state or 'vol_cols' not in st.session_state or 'model' not in st.session_state:
        st.error("Please navigate to 'Setup and Baseline' page first to load data and model.")
        return
    
    X_test = st.session_state.X_test
    vol_cols = st.session_state.vol_cols
    model = st.session_state.model # Need the model for executing scenarios later

    st.markdown("## 8. Implementing the Stress Transformation Function (`stress_scenario_volatility`)")
    st.markdown("""
    The core of our robustness test is the ability to transform input features to simulate stress. We'll implement a versatile function, `stress_scenario_volatility`, that allows us to multiply specified columns by a chosen factor. This simulates 'volatility shocks' such as sudden economic upturns or downturns.
    """)

    def stress_scenario_volatility(X_data, factor, vol_cols):
        stressed_df = X_data.copy(deep=True)
        if vol_cols is None or len(vol_cols) == 0:
            target_cols = stressed_df.select_dtypes(include=np.number).columns
        else:
            target_cols = vol_cols
        if len(target_cols) > 0:
            stressed_df[target_cols] = stressed_df[target_cols] * factor
        return stressed_df

    st.code("""\
def stress_scenario_volatility(X_data, factor, vol_cols):
    """\
    Applies a volatility shock to specified columns of a Pandas DataFrame.
    Arguments:
    X_data: Pandas DataFrame, the input features to be stressed.
    factor: float, the multiplicative factor.
    vol_cols: list of strings, names of columns to stress. If None or empty, all numerical columns.
    Output:
    Pandas DataFrame, with specified columns modified by the factor.
    """\
    stressed_df = X_data.copy(deep=True)
    if vol_cols is None or len(vol_cols) == 0:
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

    st.markdown("## 14. Customizing Stress Scenarios (Advanced)")
    st.markdown("""
    The flexibility of this framework allows for defining various types of stress beyond simple multiplicative factors. Users can implement custom functions to simulate more complex real-world phenomena, such as adding random noise, applying thresholds, or simulating specific market event impacts.
    """)

    def stress_scenario_add_noise(X_data, noise_std_dev, vol_cols):
        X_stressed = X_data.copy(deep=True)
        if vol_cols is None or len(vol_cols) == 0:
            target_cols = X_stressed.select_dtypes(include=np.number).columns
        else:
            target_cols = vol_cols
        for col in target_cols:
            noise = np.random.normal(loc=0, scale=noise_std_dev, size=len(X_stressed))
            X_stressed[col] = X_stressed[col] + noise
        return X_stressed

    st.code("""\
def stress_scenario_add_noise(X_data, noise_std_dev, vol_cols):
    """    This function introduces random Gaussian noise to specified columns of a Pandas DataFrame, simulating unpredictable measurement errors or market fluctuations. It creates a deep copy of the input DataFrame to avoid modifying the original data, preserving the original test set for baseline comparisons.
Arguments:
X_data: Pandas DataFrame, the input features to be stressed.
noise_std_dev: float, the standard deviation of the Gaussian noise to add (mean is 0).
vol_cols: list of strings, the names of the columns to add noise to. If None or empty, noise is added to all numerical columns in the DataFrame.
Output:
Pandas DataFrame, the DataFrame with specified columns having added random Gaussian noise.
    """\
    X_stressed = X_data.copy(deep=True)
    if vol_cols is None or len(vol_cols) == 0:
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

    # Sidebar for adding custom scenarios (moving this to the main area of page2 for better flow)
    st.subheader("Define Custom Stress Scenario")
    with st.container(border=True):
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
    
    st.write("Current active scenarios:")
    st.dataframe(pd.DataFrame(st.session_state.scenarios).T)

    st.markdown("We've defined three distinct scenarios: a general 'vol_up' shock across all designated volatility columns, a 'vol_down' shock, and a targeted 'single_col_spike' affecting only the `MedInc` (Median Income) column. These scenarios simulate different types of potential real-world events that could impact the model's inputs.")

    # Execution Control button
    st.markdown("## Execution Control")
    st.markdown("Click the button below to run all defined stress scenarios and generate predictions.")

    # Map function names to actual functions (these need to be defined at the module level or passed)
    # For simplicity and to avoid circular imports, define them here or make them available globally if needed.
    scenario_functions = {
        'stress_scenario_volatility': stress_scenario_volatility,
        'stress_scenario_add_noise': stress_scenario_add_noise
    }

    def execute_scenarios(model, X_test, scenarios_dict):
        stressed_predictions = {}
        for name, details in scenarios_dict.items():
            func = scenario_functions.get(details['func_name'])
            if func:
                # Ensure the correct function signature is used for params
                # Filter params based on what the function expects
                if details['func_name'] == 'stress_scenario_volatility':
                    func_params = {k: v for k, v in details['params'].items() if k in ['factor', 'vol_cols']}
                elif details['func_name'] == 'stress_scenario_add_noise':
                    func_params = {k: v for k, v in details['params'].items() if k in ['noise_std_dev', 'vol_cols']}
                else:
                    func_params = details['params'] # Fallback for other potential functions
                
                X_stress = func(X_test.copy(), **func_params)
                y_hat_stress = model.predict(X_stress)
                stressed_predictions[name] = y_hat_stress
        return stressed_predictions

    if st.button("Run Robustness Test"):
        if 'model' not in st.session_state or 'X_test' not in st.session_state or 'y_hat_base' not in st.session_state:
            st.error("Please ensure the model and baseline predictions are loaded on 'Setup and Baseline' page.")
            return
        
        with st.spinner("Running scenarios and generating predictions..."):
            st.session_state.stressed_predictions = execute_scenarios(model, X_test, st.session_state.scenarios)
            
            # Recalculate mean shifts after running scenarios
            y_hat_base = st.session_state.y_hat_base
            mean_shifts = {}
            for name, y_hat_stress in st.session_state.stressed_predictions.items():
                shift = float(y_hat_stress.mean() - y_hat_base.mean()) # Simplified calculate_mean_shift inline
                mean_shifts[name] = shift
            st.session_state.mean_shifts = mean_shifts
            
        st.success("Robustness test completed! Navigate to 'Results and Analysis' to view.")
"""))