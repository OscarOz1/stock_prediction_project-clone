# save this as app.py
from flask import Flask, escape, request, render_template
import pandas as pd
import numpy as np
import pickle
import joblib

app = Flask(__name__)

# ---------------------------------------------------------------------------------------------------------------------#
# ---------------------------------------------- ML Model Code --------------------------------------------------------#
# ---------------------------------------------------------------------------------------------------------------------#

@app.route('/')
@app.route('/about')
def about():

    return render_template("about.html")

@app.route('/resume')
def resume():

    return render_template("resume.html")

@app.route('/other_projects')
def other_projects():

    return render_template("other_projects.html")

@app.route('/stock_predictor')
def stock_predictor():

    return render_template("stock_predictor.html")

def prepDataAndPredict(high, low, bollinger_std, stochastic_lowest_low, stochastic_highest_high, open_lag_2):
    # keep all inputs in array
    data = np.array([high, low, bollinger_std, stochastic_lowest_low, stochastic_highest_high, open_lag_2])

    # Reshape array for the model (we used scaled values)
    data = data.reshape(1, -1)

    # Load the model and scaler
    with open('saved_model.pkl', 'rb') as f:
        loaded_model = pickle.load(f)
        loaded_scaler = pickle.load(f)

    # Preprocess new data
    scaled_data = loaded_scaler.transform(data)

    # predict
    pred_diff = loaded_model.predict(scaled_data)

    return round(pred_diff[0], 6)

@app.route('/predict', methods=['GET', 'POST'])
def predict_next_day():
    if request.method == "POST":
        # get form data
        high_price = request.form.get('high_price')
        low_price = request.form.get('low_price')
        try:
            closing_price = float(request.form.get('closing_price'))  # Convert to float for the addition below
        except ValueError:
            return "Please Enter valid value for the closing price."
        bollinger_std = request.form.get('bollinger_std')
        stochastic_lowest_low = request.form.get('stochastic_lowest_low')
        stochastic_highest_high = request.form.get('stochastic_highest_high')
        open_lag_2 = request.form.get('open_lag_2')

        # call preprocessDataAndPredict and pass inputs
        try:
            pred_diff = prepDataAndPredict(high_price, low_price, bollinger_std, stochastic_lowest_low, stochastic_highest_high, open_lag_2)
        except ValueError:
            return "Please Enter valid values for all your features."

        # Calculation and Categorization
        prediction = round(pred_diff + closing_price, 4)
        price_diff = abs(prediction - closing_price)

        tolerance = 0.001  # 0.1% 
        tolerance_level = closing_price * tolerance
        trend = ""

        if price_diff <= tolerance_level:
            trend = 'Flat'
        elif prediction > closing_price:
            trend = 'Uptrend'
        else:
            trend = 'Downtrend'

        # Render with additional trend information
        return render_template('predict.html', prediction=prediction, trend=trend, closing_price=closing_price) 

        pass
    pass


# Run on Correct Port
if __name__ == "__main__":
    app.run()
