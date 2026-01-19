import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('housing.csv')

X = df[[
    "Avg. Area Income",
    "Avg. Area House Age",
    "Avg. Area Number of Rooms",
    "Avg. Area Number of Bedrooms",
    "Area Population"
]]
y = df["Price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=20, random_state=42
)

pipeline = Pipeline([
    ("scalar", StandardScaler()),
    ("model", LinearRegression())
])

pipeline.fit(X_train, y_train)

predictions = pipeline.predict(X_test)

print(f"Mean Absolute Error: ${mean_absolute_error(y_test, predictions):,.2f}")
print(f"R-squared Score: {r2_score(y_test, predictions):.2f}")

joblib.dump(pipeline, "house_price_model.pkl")
print("Model saved as house_price_model.pkl")