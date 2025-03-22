import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import re

class DataPreprocessor:
    def __init__(self, target_column):
        self.target_column = target_column
        self.column_transformer = None
        self.cat_features = None
        self.num_features = None

    def fit_transform(self, data):
        df = data.copy()

        if self.target_column in df.columns:
            df.drop([target_name], axis=1, inplace=True)
        self.cat_features = df.select_dtypes(include='object').columns.tolist()
        self.num_features = df.select_dtypes(exclude='object').columns.tolist()


        """Fit and transform the data using OneHotEncoding for categorical and leaving numerical data unchanged."""
        self.column_transformer = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), self.cat_features),
                ('num', 'passthrough', self.num_features)
            ],
            remainder='passthrough'
        )
        transformed_data = self.column_transformer.fit_transform(df)
        # Creating a DataFrame with appropriate column names
        columns = self.column_transformer.get_feature_names_out()
        self.encoded_feature_names = columns
        return pd.DataFrame(transformed_data, columns=columns)

    def parse_feature_string(self, feature_string):
        """Extract feature, operator, and value from a string."""
        match = re.match(r"(.+)\s*(<=|>=|<|>)\s*(.+)", feature_string)
        if match:
            feature = match.group(1)
            operator = match.group(2)
            value = match.group(3)
            return feature, operator, value
        else:
            raise ValueError("Input string is not in the expected format.")

    def decode_feature(self, feature, value, operator="<="):
        """Decode a single feature to its original representation based on the tree's split."""
        if feature in self.encoded_feature_names:
            if 'cat__' in feature:
                original_feature = feature.split('__')[0].replace('cat__', '')
                category = feature.split('__')[-1]
                condition = f"is {category}" if (operator in ["<=", "<"] and float(value) <= 0.5) else f"is not {category}"
                return f"{condition}"
            else:
                feature = feature.split('__')[0].replace('num__', '')
                return f"{feature} {operator} {value}"
        return "Feature not recognized"

    def inverse_transform(self, data):
        """Decode each feature condition string in a DataFrame or Series."""
        for idx, row in enumerate(data):
            if 'Class' not in row: 
                feature, operator, value = self.parse_feature_string(row)
                decoded_feature = self.decode_feature(feature.replace(" ", ""), float(value), operator)
                data.iloc[idx] = decoded_feature
        return data

        
    
# Example Usage
#data = pd.DataFrame({
#    'color': ['red', 'blue', 'green', 'blue', 'red'],
#    'size': ['S', 'M', 'L', 'S', 'M'],
#    'price': [10, 15, 20, 10, 15]
#})

#preprocessor = DataPreprocessor(categorical_features=['color', 'size'], numerical_features=['price'])
#transformed_df = preprocessor.fit_transform(data)
#print(transformed_df)

# Example of decoding a feature
#print(preprocessor.decode_feature('cat__color_blue', 0.5))

data = pd.read_csv('datasets/adult_mini.csv')
print(data.info())
target_name = "income"
preprocessor = DataPreprocessor(target_name)

# Transform the data
transformed_df = preprocessor.fit_transform(data)
transformed_df.to_csv('adult_mini_transformed.csv', index=False)
print(transformed_df.info())

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(transformed_df, data[target_name], test_size=0.2, random_state=42)

# Initialize the RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Fit the model on the training data
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Calculate and print the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")


print(preprocessor.decode_feature('cat__native-country_Germany', 0.5, "<"))