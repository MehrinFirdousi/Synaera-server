import pickle
import numpy as np
import werkzeug
import flask
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

@app.route('/sendimg', methods=['POST'])
def get_image():
    imagefile = flask.request.files['image']
    filename = werkzeug.utils.secure_filename(imagefile.filename)
    print("\nReceived image File name : " + imagefile.filename)
    imagefile.save("frames/" + filename)
    return "Image Uploaded Successfully"

@app.route('/predict', methods=['POST'])
def get_prediction():
    # Works only for a single sample
    if request.method == 'POST':
        data = request.get_json()  # Get data posted as a json
        data = np.array(data)[np.newaxis, :]  # converts shape from (4,) to (1, 4)
        prediction = model.predict(data)  # runs globally loaded model on the data
    return str(prediction[0])

# @app.route('/postname/<name>', methods=['POST'])
# def print_name(name):
#    return "my name is " + name

if __name__ == '__main__':
    load_model()  # load model at the beginning once only
    app.run(host='0.0.0.0', port=80, debug=True)
