import pandas as pd
import pickle

model = pickle.load(open("model.pkl", "rb"))

sample_data = pd.DataFrame([{
    "longitude": -122.23,
    "latitude": 37.88,
    "housing_median_age": 41,
    "total_rooms": 880,
    "total_bedrooms": 129,
    "population": 322,
    "households": 126,
    "median_income": 8.3252,
    "ocean_proximity": "NEAR BAY"
}])

result = model.predict(sample_data)

print("Predicted house price: {:.2f}".format(result[0]))