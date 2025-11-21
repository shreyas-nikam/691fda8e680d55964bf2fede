Here's a comprehensive `README.md` for your Streamlit application lab project:

---

# QuLab - Scenario-Based Model Robustness Test

![QuLab Logo](https://www.quantuniversity.com/assets/img/logo5.jpg)

## Project Title and Description

This Streamlit application, "QuLab - Scenario-Based Model Robustness Test," is designed as an interactive lab environment to explore and evaluate the robustness of machine learning models. Understanding model robustness—how well a model performs under varied, often adverse, conditions—is paramount, particularly in sensitive domains like finance where unexpected market shifts can lead to significant impacts.

The application guides users through a structured process to define custom stress scenarios, apply these transformations to input data, generate new predictions, and quantitatively and visually assess the impact on model stability. By leveraging this tool, users can gain critical insights into potential vulnerabilities of their models and inform better decisions regarding model deployment, monitoring strategies, and risk management.

### Robustness Concept

Given a model $\hat{y} = f_\theta(x)$, we wish to evaluate its stability under stress transformations $T_s$ applied to inputs (e.g., volatility shocks). For each scenario $s$:
$$x^{(s)} = T_s(x), \quad \hat{y}^{(s)} = f_\theta(x^{(s)}).$$
We compare distributions of $\hat{y}^{(s)}$ to baseline predictions.

### Learning Goals

Upon completion of using this application, users will be able to:
*   Accept a pre-trained machine learning model and a test dataset as input.
*   Implement a function to apply user-defined stress transformations to specified input columns.
*   Define and execute multiple stress scenarios, comparing their prediction distributions to a baseline.
*   Calculate and display quantitative metrics, such as the mean shift in predictions, for each scenario.
*   Interpret the results of robustness tests and understand their implications for model design, monitoring, and operational constraints, particularly in contexts like finance.

## Features

The QuLab application provides the following key features:

*   **Dataset Loading & Preparation**: Automatically loads the California Housing dataset, splits it into training and testing sets, and identifies "volatility columns" for stress testing.
*   **Model Training**: Trains a simple `LinearRegression` model as a demonstration, which serves as the pre-trained model for robustness testing.
*   **Baseline Predictions**: Generates and visualizes the distribution of predictions on the original, unstressed data.
*   **Custom Stress Transformation Functions**:
    *   `stress_scenario_volatility`: Applies a multiplicative factor to specified input columns to simulate volatility shocks.
    *   `stress_scenario_add_noise`: Introduces Gaussian noise to specified columns, simulating measurement errors or random fluctuations.
*   **Interactive Scenario Definition**: Users can define and customize multiple stress scenarios via the Streamlit interface, specifying transformation types, parameters, and target columns.
*   **Scenario Execution**: Executes all defined stress scenarios, generating new predictions for each stressed dataset.
*   **Quantitative Analysis**: Calculates and displays quantitative metrics, specifically the mean shift in predictions for each scenario compared to the baseline.
*   **Visual Analysis**: Provides interactive plots (histograms and bar charts) to visualize:
    *   The distribution of baseline predictions.
    *   Overlaid distributions of stressed predictions against baseline predictions for each scenario.
    *   A consolidated bar chart of mean shifts across all scenarios.
*   **Multi-Page Navigation**: Organizes the workflow into logical sections: "Application Overview & Setup", "Scenario Definition & Baseline Analysis", and "Execution & Results".

## Getting Started

Follow these instructions to get a copy of the project up and running on your local machine.

### Prerequisites

You will need the following installed:
*   Python 3.8+
*   `pip` (Python package installer)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/quolab-model-robustness.git
    cd quolab-model-robustness
    ```
    *(Replace `https://github.com/your-username/quolab-model-robustness.git` with the actual repository URL)*

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    ```

3.  **Activate the virtual environment:**
    *   **On Windows:**
        ```bash
        .\venv\Scripts\activate
        ```
    *   **On macOS/Linux:**
        ```bash
        source venv/bin/activate
        ```

4.  **Install dependencies:**
    Create a `requirements.txt` file in the root directory of your project with the following content:
    ```
    streamlit
    pandas
    numpy
    scikit-learn
    plotly
    ```
    Then, install them using pip:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

To run the Streamlit application:

1.  **Activate your virtual environment** (if not already active):
    *   Windows: `.\venv\Scripts\activate`
    *   macOS/Linux: `source venv/bin/activate`

2.  **Run the Streamlit application** from the project's root directory:
    ```bash
    streamlit run app.py
    ```

    Your web browser should automatically open to the Streamlit application, usually at `http://localhost:8501`.

### Navigating the Application

The application is structured into three main pages, accessible via the sidebar navigation:

1.  **Application Overview & Setup**:
    *   Provides an overview of the lab and its learning objectives.
    *   Handles the loading and preparation of the California Housing dataset.
    *   Trains a `LinearRegression` model for demonstration purposes.

2.  **Scenario Definition & Baseline Analysis**:
    *   Generates and displays descriptive statistics and a distribution plot for baseline model predictions.
    *   Introduces the core stress transformation functions (`stress_scenario_volatility`, `stress_scenario_add_noise`).
    *   Allows users to define, customize, and view active stress scenarios, including adding new ones via an expander in the sidebar.

3.  **Execution & Results**:
    *   **Crucially, click the "Run Robustness Test" button on this page to execute all defined scenarios and generate results.**
    *   Displays descriptive statistics for stressed predictions.
    *   Presents calculated mean shifts (quantitative impact) for all scenarios.
    *   Visualizes prediction distributions (stressed vs. baseline) for each scenario using overlaid histograms.
    *   Provides a consolidated bar chart showing mean shifts across all scenarios for easy comparison.

## Project Structure

The project is organized as follows:

```
quolab-model-robustness/
├── app.py
├── application_pages/
│   ├── __init__.py
│   ├── page1.py
│   ├── page2.py
│   └── page3.py
├── requirements.txt
└── README.md
```

*   `app.py`: The main entry point for the Streamlit application, handling global configurations and page navigation.
*   `application_pages/`: Contains the logic and UI for each individual page of the Streamlit application.
    *   `page1.py`: Manages initial setup, data loading, and model training.
    *   `page2.py`: Handles baseline predictions, defines stress functions, and allows interactive scenario definition.
    *   `page3.py`: Orchestrates the execution of scenarios, calculates metrics, and visualizes the results.
*   `requirements.txt`: Lists all Python dependencies required to run the application.
*   `README.md`: This file, providing an overview and instructions for the project.

## Technology Stack

*   **Python**: The core programming language.
*   **Streamlit**: For building interactive web applications and user interfaces.
*   **Pandas**: Essential for data manipulation and analysis.
*   **NumPy**: For numerical operations, especially array manipulations.
*   **Scikit-learn**: Provides machine learning tools, including datasets (`fetch_california_housing`), model selection utilities (`train_test_split`), and a regression model (`LinearRegression`).
*   **Plotly**: For creating rich, interactive data visualizations (histograms, bar charts).

## Contributing

Contributions are welcome! If you have suggestions for improvements, bug fixes, or new features, please feel free to:

1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/AmazingFeature`).
3.  Make your changes.
4.  Commit your changes (`git commit -m 'Add some AmazingFeature'`).
5.  Push to the branch (`git push origin feature/AmazingFeature`).
6.  Open a Pull Request.

Please ensure your code adheres to good practices and includes appropriate documentation and tests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For any questions, suggestions, or feedback, please reach out via:

*   **GitHub Issues**: [https://github.com/your-username/quolab-model-robustness/issues](https://github.com/your-username/quolab-model-robustness/issues)
*   **Email**: [your.email@example.com](mailto:your.email@example.com)

---