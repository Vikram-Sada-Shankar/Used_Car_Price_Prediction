from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import traceback

app = Flask(__name__)

# Load trained models
try:
    RandomForest_model = joblib.load('RandomForestModel.pkl')
    XGBoost_model = joblib.load('XGBoostModel.pkl')
except Exception as e:
    print("Error loading models:", e)
    RandomForest_model = None
    XGBoost_model = None

# Load car dataset
try:
    car = pd.read_csv('cars.csv')
except Exception as e:
    print("Error loading dataset:", e)
    car = pd.DataFrame()

@app.route('/')
def index():
    if car.empty:
        return "Error: Car dataset could not be loaded."
    
    companies = sorted(car['brand'].dropna().unique())
    car_models = {brand: sorted(car[car['brand'] == brand]['model'].dropna().unique()) for brand in companies}
    seller_type = sorted(car['seller_type'].dropna().unique())
    fuel_type = sorted(car['fuel_type'].dropna().unique())
    transmission_type = sorted(car['transmission_type'].dropna().unique())

    return render_template('index.html', companies=companies, car_models=car_models, 
                           seller_type=seller_type, fuel_type=fuel_type,
                           transmission_type=transmission_type)

@app.route("/predict", methods=['POST'])
def predict():
    try:
        if not RandomForest_model or not XGBoost_model:
            return jsonify({"error": "Models could not be loaded."})

        brand = request.form.get('company')
        model = request.form.get('car_model')
        vehicle_age = int(request.form.get('vehicle_age'))
        km_driven = int(request.form.get('kilometre_driven'))
        seller_type = request.form.get('seller_type')
        fuel_type = request.form.get('fuel_type')
        transmission_type = request.form.get('transmission_type')
        mileage = float(request.form.get('mileage'))
        engine = int(request.form.get('engine'))
        max_power = float(request.form.get('max_power'))
        seats = int(request.form.get('number_of_seats'))
        depreciation_rate = request.form.get('depreciation_rate')
        model_choice = request.form.get('model_choice')

        try:
            depreciation_rate = float(depreciation_rate)
        except (TypeError, ValueError):
            depreciation_rate = 12.0  # Default value

        input_data = pd.DataFrame([[brand, model, vehicle_age, km_driven, seller_type, 
                                    fuel_type, transmission_type, mileage, engine, max_power, seats]], 
                                  columns=['brand', 'model', 'vehicle_age', 'km_driven', 
                                           'seller_type', 'fuel_type', 'transmission_type', 
                                           'mileage', 'engine', 'max_power', 'seats'])

        predictions = {}
        if model_choice == "RandomForest":
            prediction = RandomForest_model.predict(input_data)[0] * 100000
            final_price = prediction * (1 - depreciation_rate / 100)
            predictions["RandomForest"] = f"{round(final_price, 2)}₹"
        elif model_choice == "XGBoost":
            prediction = XGBoost_model.predict(input_data)[0] * 100000
            final_price = prediction * (1 - depreciation_rate / 100)
            predictions["XGBoost"] = f"{round(final_price, 2)}₹"
        else:
            prediction_1 = RandomForest_model.predict(input_data)[0] * 100000
            prediction_2 = XGBoost_model.predict(input_data)[0] * 100000
            final_price_1 = prediction_1 * (1 - depreciation_rate / 100)
            final_price_2 = prediction_2 * (1 - depreciation_rate / 100)
            predictions["RandomForest"] = f"{round(final_price_1, 2)}₹"
            predictions["XGBoost"] = f"{round(final_price_2, 2)}₹"

        return jsonify(predictions)

    except Exception as e:
        return jsonify({"error": str(traceback.format_exc())})

if __name__ == '__main__':
    app.run(debug=True)