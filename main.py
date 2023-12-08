import numpy as np
from flask import Flask, render_template, request
import io
import xgboost as xgb

print(xgb.__version__)

# initialize our Flask application and the XGB model
app = Flask(__name__)
model = xgb.XGBRegressor()
model.load_model("XGBModel.json")

@app.route("/", methods=["GET"])
def main():
    return render_template('index.html')

@app.route("/prediksi", methods=["POST"])
def predict():
    prediction = 0
    if request.method == 'POST':
        store = request.form['store']
        dept = request.form['dept']
        temperature = request.form['temperature']
        fuel_price = request.form['fuel_price']
        cpi = request.form['CPI']
        unemployment = request.form['unemployment']
        type = request.form['type']
        size = request.form['size']
        month = request.form['month']

        prediction = model.predict([[int(store),int(dept),float(temperature),float(fuel_price),float(cpi),float(unemployment),int(type),int(size),int(month)]])

    return render_template('result.html',prediction=str(prediction[0]))

# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
        "please wait until server has fully started"))
    app.run()


