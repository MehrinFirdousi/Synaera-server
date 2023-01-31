import pickle
import numpy as np
from flask import Flask, request

model = None
app = Flask(__name__)

def load_model():
	global model
	with open('iris_trained_model.pkl', 'rb') as f:
		model = pickle.load(f);

@app.route('/')
def home_endpoint():
    return 'Hello World!'

# @app.route('/predict', methods=['GET'])
# def temp_endpoint():
# 	return 'test'

@app.route('/predict', methods=['POST'])
def get_prediction():
    # Works only for a single sample
    if request.method == 'POST':
        data = request.get_json()  # Get data posted as a json
        # print(data)
        data = np.array(data)[np.newaxis, :]  # converts shape from (4,) to (1, 4)
        # print(data)
        prediction = model.predict(data)  # runs globally loaded model on the data
    return str(prediction[0])

# @app.route('/postname/<name>', methods=['POST'])
# def print_name(name):
#    return "my name is " + name

if __name__ == '__main__':
    load_model()  # load model at the beginning once only
    app.run(host='0.0.0.0', port=80)