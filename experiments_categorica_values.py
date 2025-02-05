import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from dpg.core import digraph_to_nx, get_dpg, get_dpg_node_metrics, get_dpg_metrics
from dpg.visualizer import plot_dpg_reg, plot_dpg


# Load the dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
column_names = [
    "age", "workclass", "fnlwgt", "education", "education_num", 
    "marital_status", "occupation", "relationship", "race", "sex", 
    "capital_gain", "capital_loss", "hours_per_week", "native_country", "income"
]
data = pd.read_csv(url, names=column_names, na_values="?", skipinitialspace=True)

# Handling missing values: Drop rows with missing data
data = data.dropna()

# Encode categorical variables
#label_encoders = {}
#for column in data.select_dtypes(include=['object']).columns:
#    if column != 'income':
#        le = LabelEncoder()
#        data[column] = le.fit_transform(data[column])
#        label_encoders[column] = le

# Encode target variable
data['income'] = data['income'].apply(lambda x: 1 if x.strip() == '>50K' else 0)

### Step 3: Split the data into training and testing sets

X = data.drop('income', axis=1)
y = data['income']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

### Step 4: Train a Random Forest model

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Initialize the model
rf = RandomForestClassifier(n_estimators=5, random_state=42)

# Train the model
rf.fit(X_train, y_train)

# Make predictions
y_pred = rf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

dot = get_dpg(X_train.values, column_names, rf, 0.01, 2)


dpg_model, nodes_list = digraph_to_nx(dot)
df_dpg = get_dpg_metrics(dpg_model, nodes_list)
df = get_dpg_node_metrics(dpg_model, nodes_list)

print(df_dpg)