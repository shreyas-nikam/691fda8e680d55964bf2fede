
import streamlit as st
st.set_page_config(page_title="QuLab", layout="wide")
st.sidebar.image("https://www.quantuniversity.com/assets/img/logo5.jpg")
st.sidebar.divider()
st.title("QuLab - Scenario-Based Model Robustness Test")
st.divider()
st.markdown("""
In this lab, we implement a scenario-based robustness test for machine learning models. The application guides users through defining stress scenarios, applying transformations to input data, generating stressed predictions, and quantitatively and visually assessing the impact on model stability.

### Learning Goals:
Upon completion of using this application, users will be able to:
*   Accept a pre-trained machine learning model and a test dataset as input.
*   Implement a function to apply user-defined stress transformations to specified input columns.
*   Define and execute multiple stress scenarios, comparing their prediction distributions to a baseline.
*   Calculate and display quantitative metrics, such as the mean shift in predictions, for each scenario.
*   Interpret the results of robustness tests and understand their implications for model design, monitoring, and operational constraints, particularly in contexts like finance.

### Robustness Concept:
Given a model $\hat{y} = f_\theta(x)$, we wish to evaluate its stability under stress transformations $T_s$ applied to inputs (e.g., volatility shocks). For each scenario $s$:
$$x^{(s)} = T_s(x), \quad \hat{y}^{(s)} = f_\theta(x^{(s)}).$$
We compare distributions of $\hat{y}^{(s)}$ to baseline predictions.
""").replace("```", "\`\`\`"))
# Your code starts here
page = st.sidebar.selectbox(label="Navigation", options=["Application Overview & Setup", "Scenario Definition & Baseline Analysis", "Execution & Results"])
if page == "Application Overview & Setup":
    from application_pages.page1 import run_page1
    run_page1()
elif page == "Scenario Definition & Baseline Analysis":
    from application_pages.page2 import run_page2
    run_page2()
elif page == "Execution & Results":
    from application_pages.page3 import run_page3
    run_page3()
# Your code ends
