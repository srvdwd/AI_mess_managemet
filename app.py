from flask import Flask, render_template, request
from model.model import CanteenModel

app = Flask(__name__)
model = CanteenModel()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    day_of_week = int(request.form['day_of_week'])
    is_event = int(request.form['is_event'])
    past_attendance = int(request.form['past_attendance'])

    prediction = model.predict_demand(day_of_week, is_event, past_attendance)
    return render_template('index.html', prediction=prediction)

@app.route('/suggest_meals')
def suggest_meals():
    meals = model.get_meal_suggestions()
    return render_template('index.html', meals=meals)

@app.route('/feedback', methods=['POST'])
def feedback():
    feedback_text = request.form['feedback']
    sentiment = model.analyze_feedback(feedback_text)
    return render_template('index.html', sentiment=sentiment)

if __name__ == '__main__':
    app.run(debug=True)
