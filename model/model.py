import pandas as pd
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from textblob import TextBlob

print("numpy", np.__version__)
print("pandas", pd.__version__)
print("sklearn", sklearn.__version__)

class CanteenModel:
    def __init__(self):
        self.demand_model = self._train_demand_model()
        self.meal_suggestions = self._generate_meal_suggestions()

    def _train_demand_model(self):
        data = pd.read_csv("canteen_data.csv")

        X = data[['day_of_week', 'is_event', 'past_attendance']]
        y = data['meals_served']

        X.fillna(X.mean(), inplace=True)
        y.fillna(y.mean(), inplace=True)

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=5)
        print(f"Cross-Validation MAE: {-np.mean(scores):.2f}")

        model.fit(X, y)

        with open("model.pkl", "wb") as f:
            pickle.dump(model, f)

        return model

    def _generate_meal_suggestions(self):
        meals = pd.DataFrame({
            'meal': ['Dal Chawal', 'Paneer Tikka', 'Salad Bowl', 'Rajma Rice', 'Mixed Veg', 'Fruit Bowl'],
            'calories': [450, 520, 200, 470, 300, 150],
            'protein': [15, 25, 5, 20, 10, 2],
            'cost': [25, 40, 15, 30, 20, 10]
        })
        meals['score'] = (meals['protein'] / meals['cost']) - (meals['calories'] / 1000)
        return meals.sort_values(by='score', ascending=False)

    def predict_demand(self, day_of_week, is_event, past_attendance):
        input_data = [[day_of_week, is_event, past_attendance]]
        return int(self.demand_model.predict(input_data)[0])

    def get_meal_suggestions(self, count=3):
        return self.meal_suggestions.head(count).to_dict('records')

    def analyze_feedback(self, feedback_text):
        return TextBlob(feedback_text).sentiment.polarity


