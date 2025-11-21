"""import streamlit as st
st.set_page_config(page_title="QuLab", layout="wide")
st.sidebar.image("https://www.quantuniversity.com/assets/img/logo5.jpg")
st.sidebar.divider()
st.title("QuLab")
st.divider()
st.markdown("""
### Scenario-Based Model Robustness Test

In this lab, we explore the robustness of machine learning models by subjecting them to various stress scenarios. Understanding model robustness is crucial in many domains, especially in finance, where models need to perform reliably even under adverse or unexpected market conditions.

The application allows users to:
- Load a dataset and a pre-trained model.
- Define custom stress transformations for input features.
- Apply these transformations to create 'stressed' versions of the input data.
- Generate predictions from the model on both baseline and stressed data.
- Quantitatively measure and visually compare the impact of stress on model predictions.

**Robustness Concept:**
Given a model $\\hat{y} = f_\\theta(x)$, we wish to evaluate its stability under stress transformations $T_s$ applied to inputs (e.g., volatility shocks). For each scenario $s$:
$$x^{(s)} = T_s(x), \\quad \\hat{y}^{(s)} = f_\\theta(x^{(s)}).$$
We compare distributions of $\\hat{y}^{(s)}$ to baseline predictions.

By exploring different scenarios, users gain insights into potential vulnerabilities and can make informed decisions about model deployment, monitoring strategies, and risk management.
""")
# Your code starts here
page = st.sidebar.selectbox(label="Navigation", options=["Setup and Baseline", "Define Scenarios", "Results and Analysis"])
if page == "Setup and Baseline":
    from application_pages.page1 import run_page1
    run_page1()
elif page == "Define Scenarios":
    from application_pages.page2 import run_page2
    run_page2()
elif page == "Results and Analysis":
    from application_pages.page3 import run_page3
    run_page3()
# Your code ends
"""))