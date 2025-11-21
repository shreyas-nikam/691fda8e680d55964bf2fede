
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Map function names to actual functions - these should be available in the context of page2 or imported if needed
# For simplicity, assuming these are defined or imported within the scope where execute_scenarios is called.
# However, for a multi-page app, it's better to centralize utility functions or re-define if not shared.
# For this task, they were provided in the prompt for page2, so we can re-define them here or import from page2.
# Given the prompt, I will assume the functions `stress_scenario_volatility` and `stress_scenario_add_noise` are available.


def stress_scenario_volatility(X_data, factor, vol_cols):
    """
    Applies a volatility shock to specified columns of a Pandas DataFrame.
    Arguments:
    X_data: Pandas DataFrame, the input features to be stressed.
    factor: float, the multiplicative factor.
    vol_cols: list of strings, names of columns to stress. If None, all numerical columns.
    Output:
    Pandas DataFrame, with specified columns modified by the factor.
    """
    stressed_df = X_data.copy(deep=True)
    if vol_cols is None:
        target_cols = stressed_df.select_dtypes(include=np.number).columns
    else:
        target_cols = vol_cols
    if len(target_cols) > 0:
        # Ensure columns exist before attempting to modify
        valid_target_cols = [
            col for col in target_cols if col in stressed_df.columns]
        if valid_target_cols:
            stressed_df[valid_target_cols] = stressed_df[valid_target_cols] * factor
    return stressed_df


def stress_scenario_add_noise(X_data, noise_std_dev, vol_cols):
    """    This function introduces random Gaussian noise to specified columns of a Pandas DataFrame, simulating unpredictable measurement errors or market fluctuations. It creates a deep copy of the input DataFrame to avoid modifying the original data, preserving the original test set for baseline comparisons.
Arguments:
X_data: Pandas DataFrame, the input features to be stressed.
noise_std_dev: float, the standard deviation of the Gaussian noise to add (mean is 0).
vol_cols: list of strings, the names of the columns to add noise to. If None, noise is added to all numerical columns in the DataFrame.
Output:
Pandas DataFrame, the DataFrame with specified columns having added random Gaussian noise.
    """
    X_stressed = X_data.copy(deep=True)
    if vol_cols is None:
        target_cols = X_stressed.select_dtypes(include=np.number).columns
    else:
        target_cols = vol_cols
    if len(target_cols) > 0:
        valid_target_cols = [
            col for col in target_cols if col in X_stressed.columns]
        for col in valid_target_cols:
            noise = np.random.normal(
                loc=0, scale=noise_std_dev, size=len(X_stressed))
            X_stressed[col] = X_stressed[col] + noise
    return X_stressed


def run_page3():
    st.title("Execution & Results")

    # Ensure necessary session_state variables are initialized
    if 'X_test' not in st.session_state or 'model' not in st.session_state or 'y_hat_base' not in st.session_state or 'scenarios' not in st.session_state:
        st.warning("Please navigate to 'Application Overview & Setup' and 'Scenario Definition & Baseline Analysis' first to set up the environment.")
        return

    X_test = st.session_state.X_test
    model = st.session_state.model
    y_hat_base = st.session_state.y_hat_base
    scenarios_dict = st.session_state.scenarios

    st.markdown(
        "## 10. Executing Stress Scenarios and Generating Stressed Predictions")
    st.markdown("""
    With our scenarios defined, we now execute each one. For every scenario, we apply the stress transformation to a copy of `X_test` and then generate new predictions using our pre-trained model. These stressed predictions will then be compared to our baseline.
    """)

    # Map function names to actual functions within this scope for execution
    scenario_functions = {
        'stress_scenario_volatility': stress_scenario_volatility,
        'stress_scenario_add_noise': stress_scenario_add_noise
    }

    def execute_scenarios(model, X_data, scenarios_dict):
        stressed_predictions = {}
        for name, details in scenarios_dict.items():
            func = scenario_functions.get(details['func_name'])
            if func:
                X_stress = func(X_data.copy(), **details['params'])
                y_hat_stress = model.predict(X_stress)
                stressed_predictions[name] = y_hat_stress
        return stressed_predictions

    if st.button("Run Robustness Test"):
        with st.spinner("Running scenarios and generating predictions..."):
            st.session_state.stressed_predictions = execute_scenarios(
                model, X_test, scenarios_dict)
        st.success("Robustness test completed!")
        st.rerun()  # Rerun to display results immediately

    stressed_predictions = st.session_state.get('stressed_predictions', {})

    if stressed_predictions:
        st.write("Scenarios processed:", list(stressed_predictions.keys()))
        if 'vol_up' in stressed_predictions:
            st.write("Descriptive statistics for 'vol_up' stressed predictions:")
            st.dataframe(pd.Series(stressed_predictions['vol_up']).describe())

    st.markdown("""
    We have successfully applied each defined stress scenario and collected the corresponding model predictions. Observing the statistics for a sample stressed scenario, like 'vol_up', already hints at how the model's output distribution might have shifted compared to the baseline.
    """)

    st.markdown("## 11. Calculating Quantitative Shift Metrics")
    st.markdown("""
    While visual comparisons are intuitive, quantitative metrics provide a concise summary of the impact. A common metric is the **mean shift** in predictions, which indicates the average change in the model's output due to the stress.
    """)

    def calculate_mean_shift(baseline_predictions, stressed_predictions):
        mean_baseline = baseline_predictions.mean()
        mean_stressed = stressed_predictions.mean()
        return float(mean_stressed - mean_baseline)

    def calculate_mean_shift(baseline_predictions, stressed_predictions):
        """
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
        """
        mean_baseline = baseline_predictions.mean()
        mean_stressed = stressed_predictions.mean()
        return float(mean_stressed - mean_baseline)

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

    st.markdown("## 12. Visualizing Prediction Distributions for Each Scenario")
    st.markdown("""
    To gain a deeper understanding of the impact of each stress scenario, we will visualize the distribution of stressed predictions against the baseline. This allows us to observe not just mean shifts, but also changes in variance, skewness, or the emergence of new modes in the predictions.
    """)

    if stressed_predictions and mean_shifts:
        for name, y_hat_stress in stressed_predictions.items():
            df_plot = pd.DataFrame({
                'Predictions': np.concatenate([y_hat_base, y_hat_stress]),
                'Type': ['Baseline'] * len(y_hat_base) + [name] * len(y_hat_stress)
            })

            fig_dist = px.histogram(df_plot, x="Predictions", color="Type", histnorm='density',
                                    title=f"Prediction Distribution: Baseline vs. {name}",
                                    color_discrete_map={
                                        'Baseline': 'blue', name: 'red'},
                                    barmode='overlay', opacity=0.6)
            fig_dist.update_layout(bargap=0.1)

            shift = mean_shifts.get(name, 0.0)
            fig_dist.add_annotation(dict(font=dict(color="black", size=12),
                                         x=0.05, y=0.9,
                                         showarrow=False,
                                         text=f'Mean Shift: {shift:.2f}',
                                         textangle=0,
                                         xanchor='left', xref="paper",
                                         yanchor='top', yref="paper"))

            st.plotly_chart(fig_dist)

    st.markdown("""
    These overlaid plots visually confirm the shifts quantified earlier. They also reveal more subtle changes, such as whether the stress scenario caused predictions to become more spread out (increased variance), skewed, or if certain prediction values became more or less common.
    """)

    st.markdown("## 13. Visualizing Quantitative Shifts (Bar Chart)")
    st.markdown("""
    A bar chart of the mean shifts provides a consolidated view, making it easy to compare the magnitude of impact across all scenarios and quickly identify the most disruptive ones.
    """)

    if mean_shifts:
        mean_shifts_series = pd.Series(mean_shifts)
        fig_bar = px.bar(mean_shifts_series, y=mean_shifts_series.values, x=mean_shifts_series.index,
                         title="Mean Shift in Predictions Across Scenarios",
                         labels={"x": "Stress Scenario", "y": "Mean Shift"},
                         color=mean_shifts_series.values, color_continuous_scale=px.colors.sequential.Viridis)
        fig_bar.add_hline(y=0, line_dash="dash", line_color="gray")
        fig_bar.update_xaxes(tickangle=45)
        st.plotly_chart(fig_bar)

    st.markdown("""
    This bar chart offers a clear, at-a-glance summary of how each stress scenario affected the average model prediction. Scenarios with larger absolute mean shifts indicate areas where the model's stability is more significantly compromised.
    """)

    st.markdown("## 15. Rerunning Analysis with Custom Scenarios")
    st.markdown("""
    To see the impact of our newly defined custom stress scenario, we will re-execute the prediction and analysis steps, incorporating the 'noise_shock'. This iterative process is crucial for thoroughly evaluating model robustness under diverse conditions.
    """)

    # The 'Run Robustness Test' button will handle this. Display updated mean shifts if available.
    if mean_shifts:
        st.write("Updated Calculated Mean Shifts (including custom scenarios):")
        st.json(mean_shifts)

    st.markdown("""
    The updated `mean_shifts` now include the impact of any newly added custom scenarios. This iterative approach allows us to dynamically assess new stress scenarios and their effects, providing a comprehensive view of the model's behavior.
    """)
