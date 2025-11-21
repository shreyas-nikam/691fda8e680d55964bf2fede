"""import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

def run_page3():
    st.title("Page 3: Results and Analysis")

    if 'y_hat_base' not in st.session_state or 'stressed_predictions' not in st.session_state or 'mean_shifts' not in st.session_state:
        st.warning("Please run the 'Robustness Test' from the 'Define Scenarios' page first to see results.")
        return

    y_hat_base = st.session_state.y_hat_base
    stressed_predictions = st.session_state.stressed_predictions
    mean_shifts = st.session_state.mean_shifts

    st.markdown("## 10. Executing Stress Scenarios and Generating Stressed Predictions")
    st.markdown("""
    With our scenarios defined and executed, we now review the generated predictions. These stressed predictions are critical for comparing against our baseline.
    """)

    if stressed_predictions:
        st.write("Scenarios processed:", list(stressed_predictions.keys()))
        if 'vol_up' in stressed_predictions:
            st.write("\nDescriptive statistics for 'vol_up' stressed predictions:")
            st.dataframe(pd.Series(stressed_predictions['vol_up']).describe())

    st.markdown("""
    We have successfully applied each defined stress scenario and collected the corresponding model predictions. Observing the statistics for a sample stressed scenario, like 'vol_up', already hints at how the model's output distribution might have shifted compared to the baseline.
    """)

    st.markdown("## 11. Calculating Quantitative Shift Metrics")
    st.markdown("""
    While visual comparisons are intuitive, quantitative metrics provide a concise summary of the impact. A common metric is the **mean shift** in predictions, which indicates the average change in the model's output due to the stress.
    """)

    st.code("""\
# The calculation of mean shift is performed when the 'Run Robustness Test' button is clicked.
# For reference, the function is:
# def calculate_mean_shift(baseline_predictions, stressed_predictions):
#     mean_baseline = baseline_predictions.mean()
#     mean_stressed = stressed_predictions.mean()
#     return float(mean_stressed - mean_baseline)
""")

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
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=y_hat_base, histnorm='density', name='Baseline', opacity=0.5, marker_color='blue'))
            fig.add_trace(go.Histogram(x=y_hat_stress, histnorm='density', name=name, opacity=0.5, marker_color='red'))
            
            shift = mean_shifts.get(name, 0.0)
            
            fig.update_layout(title_text=f"Prediction Distribution: Baseline vs. {name}",
                              xaxis_title="Predicted Value",
                              yaxis_title="Density",
                              annotations=[go.layout.Annotation(
                                  xref="paper", yref="paper",
                                  x=0.05, y=0.9,
                                  text=f'Mean Shift: {shift:.2f}',
                                  showarrow=False,
                                  bgcolor="rgba(255,255,255,0.7)",
                                  bordercolor="black",
                                  borderwidth=1
                              )])
            st.plotly_chart(fig)

    st.markdown("""
    These overlaid plots visually confirm the shifts quantified earlier. They also reveal more subtle changes, such as whether the stress scenario caused predictions to become more spread out (increased variance), skewed, or if certain prediction values became more or less common.
    """)

    st.markdown("## 13. Visualizing Quantitative Shifts (Bar Chart)")
    st.markdown("""
    A bar chart of the mean shifts provides a consolidated view, making it easy to compare the magnitude of impact across all scenarios and quickly identify the most disruptive ones.
    """)

    if mean_shifts:
        mean_shifts_df = pd.DataFrame(list(mean_shifts.items()), columns=['Scenario', 'Mean Shift'])
        fig = px.bar(mean_shifts_df, x='Scenario', y='Mean Shift',
                     title="Mean Shift in Predictions Across Scenarios",
                     color='Mean Shift', color_continuous_scale=px.colors.sequential.Viridis)
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig)

    st.markdown("""
    This bar chart offers a clear, at-a-glance summary of how each stress scenario affected the average model prediction. Scenarios with larger absolute mean shifts indicate areas where the model's stability is more significantly compromised.
    """)
    
    st.markdown("## 15. Rerunning Analysis with Custom Scenarios")
    st.markdown("""
    The updated `mean_shifts` now include the impact of any newly added custom scenarios. This iterative approach allows us to dynamically assess new stress scenarios and their effects, providing a comprehensive view of the model's behavior.
    """)
    # The actual rerunning is triggered from page2, this page just displays the updated results.
    if mean_shifts:
        st.write("Updated Calculated Mean Shifts (including custom scenarios):")
        st.json(mean_shifts)
"""))