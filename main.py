import pickle
from flask import Flask, request, jsonify, render_template
from model.ml_model import predict_mpg

app = Flask('app')

@app.route('/home', methods=['GET'])
def test():
    return render_template('home.html')

@app.route('/predict', methods=['GET'])
def predict():
    # vehicle = request.get_json()
    # print(vehicle)

    vehicle= {
        'Cylinders': [float(request.args.get('cylinders'))],
        'Displacement': [float(request.args.get('displacement'))],
        'Horsepower': [float(request.args.get('horsepower'))],
        'Weight': [float(request.args.get('weight'))],
        'Acceleration': [float(request.args.get('acceleration'))],
        'Model Year': [int(request.args.get('model-year'))],
        'Origin': [int(request.args.get('origin'))],
    }

    with open('./model/model.bin', 'rb') as f_in:
        model = pickle.load(f_in)
        f_in.close()
    predictions = predict_mpg(vehicle, model)
    predictions = [round(x,2) for x in predictions]

    result = {
        'Miles Per Gallon Prediction': list(predictions)
    }

    return render_template('output.html', mpg=result)

#for development, uncomment the session below 
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)