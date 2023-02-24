import numpy as np
import werkzeug
import flask
import keras.models
import os
import cv2
import mediapipe as mp
from PIL import Image
from flask import Flask, request

app = Flask(__name__)

global model
global frame_rate
global sequence
global mp_holistic
global mp_drawing
global actions
global colors

frame_rate = 25
sequence = []
mp_holistic = mp.solutions.holistic # Holistic model - make our detection
mp_drawing = mp.solutions.drawing_utils # Drawing utilities - make our drawings
actions = np.array(['NoSign','hello', 'thanks', 'iloveyou'])
colors = [(245,117,16), (117,245,16), (16,117,245),(16,117,245)]
model = keras.models.load_model('test8.h5')

@app.route('/')
def home_endpoint():
    return 'Hello World!'
# -------------------------------------- FUNCTIONS -------------------------------------:

# To extract keypoint values from frame using mediapipe
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results

# To draw landmarks and pose connections on the frame using the results extracted
def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION) # Draw face connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS) # Draw pose connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw right hand connections

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        
    return output_frame

def run_model(frame, frame_no):
    # sequence = []
    predictions = []
    result_p = "nothing"
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        # Make detections
        image, results = mediapipe_detection(frame, holistic)
        
        # Draw landmarks
        draw_landmarks(image, results)

        # Prediction logic
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        last_frames = sequence[-25:]
        
        # if len(last_frames) == 25:
        if ((int(frame_no) % 25 == 0) and (len(last_frames) == 25)):
            res = model.predict(np.expand_dims(last_frames, axis=0))[0]
            print(actions[np.argmax(res)])
            result_p = actions[np.argmax(res)]
            predictions.append(np.argmax(res))
    return (result_p)

@app.route('/sendImg/<frameCount>', methods=['GET', 'POST'])
def get_image(frameCount):
    image_str = flask.request.files['image'].read()
    image_bytes = np.fromstring(image_str, np.uint8)
    frame = cv2.imdecode(image_bytes, cv2.IMREAD_UNCHANGED)
    print("\nReceived", frameCount, "frames. len(seq)=", len(sequence))
    return(run_model(frame, frameCount))

@app.route('/predict', methods=['POST'])
def get_prediction():
    # Works only for a single sample
    if request.method == 'POST':
        data = request.get_json()  # Get data posted as a json
        data = np.array(data)[np.newaxis, :]  # converts shape from (4,) to (1, 4)
        prediction = model.predict(data)  # runs globally loaded model on the data
    return str(prediction[0])

@app.route('/test/<name>', methods=['GET', 'POST'])
def test_endpoint(name):
    return name

if __name__ == '__main__':
    # load_model()  # load model at the beginning once only
    app.run(host='0.0.0.0', port=5000)
