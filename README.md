# Machine Learning-Based Concrete Compressive Strength Prediction

This project uses machine learning to predict the compressive strength of concrete based on its mixture composition and curing age.

Concrete compressive strength is an important property in civil engineering because it helps determine whether a concrete mixture is suitable for construction applications. Instead of relying only on laboratory testing, this project explores how machine learning can estimate concrete strength from measurable material and mixture characteristics.

## Project Overview

The project is implemented in a Jupyter Notebook and uses the `Concrete_Data.csv` dataset. The dataset contains concrete mixture information, including cement, slag, fly ash, water, superplasticizer, aggregates, and concrete age.

The target variable is the concrete compressive strength, represented by:

```text
fck (MPa)
```

The notebook contains the data analysis and machine learning workflow for studying the relationship between concrete composition and compressive strength.

## Objectives

The main objectives of this project are to:

- Load and inspect the concrete strength dataset.
- Understand the structure and characteristics of the data.
- Analyze the relationship between concrete ingredients and compressive strength.
- Prepare the data for machine learning.
- Train regression models to predict concrete compressive strength.
- Evaluate model performance using suitable regression metrics.
- Compare predicted strength values with actual laboratory measurements.
- Visualize important trends and prediction results.

## Dataset

The dataset is stored in:

```text
Concrete_Data.csv
```

It contains concrete mixture records with the following variables.

| Feature | Description | Unit |
|---|---|---|
| Cement | Quantity of cement in the concrete mixture | kg/m³ |
| Blast Furnace Slag | Quantity of blast furnace slag | kg/m³ |
| Fly Ash | Quantity of fly ash | kg/m³ |
| Water | Quantity of water | kg/m³ |
| Superplasticizer | Quantity of superplasticizer | kg/m³ |
| Coarse Aggregate | Quantity of coarse aggregate | kg/m³ |
| Fine Aggregate | Quantity of fine aggregate | kg/m³ |
| Age | Age of the concrete sample | days |
| `fck` | Concrete compressive strength, used as the target | MPa |

### Input Features

The prediction model uses the following input features:

```text
Cement
Blast Furnace Slag
Fly Ash
Water
Superplasticizer
Coarse Aggregate
Fine Aggregate
Age
```

### Target Variable

The target variable is:

```text
fck (MPa)
```

This represents the compressive strength of the concrete measured in megapascals.

## Repository Structure

```text
ML-Compressive-Strength-prediction/
│
├── Concrete_Data.csv
├── ML_Compressive_Strength_prediction.ipynb
├── README.md
├── LICENSE
└── .gitignore
```

### File Descriptions

- `Concrete_Data.csv`  
  Dataset containing concrete mixture properties and measured compressive strength.

- `ML_Compressive_Strength_prediction.ipynb`  
  Jupyter Notebook containing the data analysis, visualizations, preprocessing, machine learning workflow, and evaluation.

- `README.md`  
  Project documentation.

- `LICENSE`  
  License information for the project.

- `.gitignore`  
  Specifies files and directories that should not be tracked by Git.

## Methodology

The project follows a typical supervised machine learning regression workflow.

### 1. Data Loading

The dataset is loaded from the CSV file using Python data analysis tools.

### 2. Data Inspection

The data is inspected to identify:

- Dataset dimensions
- Column names
- Data types
- Missing values
- Duplicate records
- Basic statistical information
- Feature distributions

### 3. Exploratory Data Analysis

Exploratory analysis is used to understand the relationships between the concrete ingredients and compressive strength.

Typical analysis includes:

- Summary statistics
- Histograms
- Box plots
- Feature distribution plots
- Correlation analysis
- Scatter plots
- Heatmaps

These visualizations help identify trends, outliers, and potentially influential features.

### 4. Feature and Target Separation

The dataset is separated into:

- `X`: input features used for prediction
- `y`: target concrete compressive strength

Conceptually:

```python
X = data.drop("fck(MPa)", axis=1)
y = data["fck(MPa)"]
```

The exact column name may require minor adjustment depending on whitespace or formatting in the CSV file.

### 5. Data Preprocessing

The data may be prepared for modeling through steps such as:

- Cleaning column names
- Converting values to numeric format
- Handling missing values
- Removing unnecessary spaces
- Checking for duplicate records
- Splitting the data into training and testing subsets
- Scaling features when required by the selected algorithm

### 6. Train-Test Split

The dataset is divided into training and testing sets.

The training data is used to fit the machine learning models, while the testing data is used to evaluate how well the models perform on unseen examples.

### 7. Regression Modeling

Because compressive strength is a continuous numerical value, the project is treated as a regression problem.

Possible regression algorithms explored in the notebook may include:

- Linear Regression
- Decision Tree Regression
- Random Forest Regression
- Gradient Boosting Regression
- Support Vector Regression
- Other regression algorithms implemented in the notebook

The exact list of models and their configurations should be taken directly from the notebook.

### 8. Model Evaluation

Regression models can be evaluated using metrics such as:

#### Mean Absolute Error

MAE measures the average absolute difference between actual and predicted values.

```text
MAE = average(|actual - predicted|)
```

Lower values indicate better performance.

#### Mean Squared Error

MSE calculates the average squared difference between actual and predicted values.

```text
MSE = average((actual - predicted)²)
```

Lower values indicate better performance.

#### Root Mean Squared Error

RMSE is the square root of the mean squared error.

```text
RMSE = √MSE
```

Lower values indicate better performance.

#### R² Score

The coefficient of determination measures how much of the variation in compressive strength is explained by the model.

```text
R² = 1 - residual sum of squares / total sum of squares
```

A higher R² score generally indicates a better fit.

## Example Prediction Workflow

A typical prediction workflow can be represented as:

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

data = pd.read_csv("Concrete_Data.csv")

data.columns = data.columns.str.strip()

X = data.drop(columns=["fck(MPa)"])
y = data["fck(MPa)"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = RandomForestRegressor(
    random_state=42
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = mean_squared_error(y_test, predictions) ** 0.5
r2 = r2_score(y_test, predictions)

print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R² Score:", r2)
```

This example is provided to explain the general workflow. The final implementation should follow the code contained in the project notebook.

## Visualizations

The notebook can be used to visualize:

- Distributions of concrete ingredients
- Distribution of compressive strength
- Correlations between features
- Feature relationships
- Actual versus predicted strength
- Prediction errors
- Model performance comparisons
- Feature importance, when supported by the model

An actual-versus-predicted plot is especially useful for assessing model quality. Predictions closer to the diagonal line generally indicate better performance.

## Interpretation

The compressive strength of concrete can be influenced by several factors, including:

- Cement content
- Supplementary cementitious materials
- Water content
- Superplasticizer quantity
- Aggregate proportions
- Concrete age

Concrete age is particularly important because strength generally develops over time as the cement hydration process continues. However, the effect of each feature depends on the overall mixture composition and the relationships learned from the dataset.

## Requirements

To run the notebook, install Python and the required packages.

Recommended packages include:

```text
Python 3.8+
Jupyter Notebook or JupyterLab
pandas
numpy
matplotlib
seaborn
scikit-learn
```

## Installation

Clone the repository:

```bash
git clone https://github.com/mohamedsoltanetrabelsi/ML-Compressive-Strength-prediction.git
```

Move into the project directory:

```bash
cd ML-Compressive-Strength-prediction
```

Create and activate a virtual environment:

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### macOS and Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

Install the required packages:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

Start Jupyter Notebook:

```bash
jupyter notebook
```

Open the following file:

```text
ML_Compressive_Strength_prediction.ipynb
```

Run the notebook cells in order.

## Reproducibility

For reproducible results:

- Use the same dataset included in the repository.
- Use the same train-test split configuration as the notebook.
- Keep random seeds consistent.
- Use compatible versions of Python and the machine learning libraries.
- Run the notebook cells in their original order.

## Limitations

This project is intended for educational and analytical purposes. Predictions should not be used as a replacement for certified laboratory testing or engineering judgment.

The model may be affected by:

- Dataset size and representativeness
- Measurement errors
- Unseen concrete mixtures
- Differences in materials and construction conditions
- Data preprocessing choices
- Model selection and hyperparameters

A model trained on this dataset may not generalize perfectly to every concrete formulation or construction environment.

## Future Improvements

Possible improvements include:

- Hyperparameter tuning
- Cross-validation
- Feature engineering
- Model ensembling
- Additional regression algorithms
- Explainable machine learning
- Error analysis
- External validation using new concrete data
- Building a web interface for interactive predictions
- Deploying the best-performing model as an API

## Notebook

The complete analysis is available in:

```text
ML_Compressive_Strength_prediction.ipynb
```

## License

This project is distributed under the license included in the `LICENSE` file.

## Author

Created by [mohamedsoltanetrabelsi](https://github.com/mohamedsoltanetrabelsi).

## Disclaimer

This project demonstrates the use of machine learning for estimating concrete compressive strength. The predictions are not intended to replace laboratory testing, structural design standards, or professional engineering advice.
