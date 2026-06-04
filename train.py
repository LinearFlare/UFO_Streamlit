import pickle
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
)


ufos = pd.read_csv("./data/ufos.csv")
print(f"Raw data shape:{ufos.shape}")
print(ufos.head())

ufos = pd.DataFrame(
    {
        "Seconds": ufos["duration (seconds)"],
        "Country": ufos["country"],
        "Latitude": ufos["latitude"],
        "Longitude": ufos["longitude"],
    }
)

print(f"\nCountries: {ufos.Country.unique()}")

ufos.dropna(inplace=True)
ufos = ufos[(ufos["Seconds"] >= 1) & (ufos["Seconds"] <= 60)]
print(f"\nCleaned data shape:{ufos.shape}")


le = LabelEncoder()
ufos["Country"] = le.fit_transform(ufos["Country"])
print(f"Country mapping: {dict(zip(le.classes_, le.transform(le.classes_)))}")

x = ufos[["Seconds", "Latitude", "Longitude"]]
y = ufos["Country"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

model = LogisticRegression(max_iter=1000)
model.fit(x_train, y_train)

predictions = model.predict(x_test)
print(f"\n{classification_report(y_test, predictions)}")
print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")

os.makedirs("model", exist_ok=True)

pickle.dump(model, open("model/ufo-model.pkl", "wb"))
print("\nModel saved to model/ufos-model.pkl")


test_pred = model.predict(
    pd.DataFrame([[50, 44, -12]], columns=["Seconds", "Latitude", "Longitude"])
)
countries = ["Australia", "Canada", "Germany", "UK", "US"]
print(f"Test prediction for [50s, lat=44, lon=-12]: {countries[test_pred[0]]}")
