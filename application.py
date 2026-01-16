from flask import Flask,request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app = application

# import ridge regressor and standard scaler
ridge_model = pickle.load(open('models/ridge.pkl', 'rb'))
scaler = pickle.load(open('models/scaler.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=["GET", "POST"])
def predict_datapoint():
    if request.method == 'POST':
        Temmperature = float(request.form['Temperature'])
        Relative_Humidity = float(request.form['RH'])
        Wind_Speed = float(request.form['Ws'])
        Rainfall = float(request.form['Rain'])
        FFMC = float(request.form['FFMC'])
        DMC = float(request.form['DMC'])
        ISI = float(request.form['ISI'])
        Classes = float(request.form['Classes'])
        Region = float(request.form['Region'])
        data = {
            'Temperature': Temmperature,
            'RH': Relative_Humidity,
            'Ws': Wind_Speed,
            'Rain': Rainfall,
            'FFMC': FFMC,
            'DMC': DMC,
            'ISI': ISI,
            'Classes': Classes,
            'Region': Region
        }

        data_df = pd.DataFrame([data])

        # Preprocess the data
        data_scaled = scaler.transform(data_df)
        
        # Make prediction
        prediction = ridge_model.predict(data_scaled)
        print(prediction)
        
        return render_template('home.html', results=float(prediction[0]))
    else:
        return render_template('home.html')

if __name__ == '__main__':
    app.run(host="0.0.0.0")