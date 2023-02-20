# import pickle
# import pandas
import numpy as np
import werkzeug
import flask
import keras.models
import os
import cv2
import mediapipe as mp
from flask import Flask, request

model = None
app = Flask(__name__)

# def load_model():
# 	global model
# 	with open('iris_trained_model.pkl', 'rb') as f:
# 		model = pickle.load(f);

def load_model():
    global model
    global frame_paths
    global mp_holistic
    global mp_drawing
    global actions
    global colors
    global pred

    frame_paths = []
    mp_holistic = mp.solutions.holistic # Holistic model - make our detection
    mp_drawing = mp.solutions.drawing_utils # Drawing utilities - make our drawings
    colors = [(245,117,16), (117,245,16), (16,117,245),(16,117,245)]
    actions = np.array(['NoSign','hello', 'thanks', 'iloveyou'])
    pred = ""
    model = keras.models.load_model(r'C:\Users\HP\Documents\Mehrin\CSIT321\nlp-model\Computer Vision Model\Notebooks\Saved Weights\test8.h5')

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

def run_model():
    sequence = []
    sentence = []
    predictions = []
    threshold = 0.5
    result_p = "nothing"
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        # for i in range(len(frame_paths)):
        for i in range(25):
            # Read feed
            frame = cv2.imread(frame_paths[i])
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            # cv2.imshow('Cat', frame)
            # cv2.waitKey(0)
            # Make detections
            image, results = mediapipe_detection(frame, holistic)
            #print(results)
            
            # Draw landmarks
            draw_landmarks(image, results)
            
            # 2. Prediction logic
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-25:]
            
            if len(sequence) == 25:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                print(actions[np.argmax(res)])
                result_p = actions[np.argmax(res)]
                predictions.append(np.argmax(res))
                
                
            #3. Viz logic
            #     if np.unique(predictions[-10:])[0]==np.argmax(res): 
            #         if res[np.argmax(res)] > threshold: 
                        
            #             if len(sentence) > 0: 
            #                 if actions[np.argmax(res)] != sentence[-1]:
            #                     sentence.append(actions[np.argmax(res)])
            #             else:
            #                 sentence.append(actions[np.argmax(res)])

            #     if len(sentence) > 5: 
            #         sentence = sentence[-5:]

            #     # Viz probabilities
            #     image = prob_viz(res, actions, image, colors)
                
            # cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
            # cv2.putText(image, ' '.join(sentence), (3,30), 
            #                 cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            # # Show to screen
            # cv2.imshow('OpenCV Feed', image)

            # # Break gracefully
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
    return (result_p)

@app.route('/sendImg', methods=['POST'])
def get_image():
    imagefile = flask.request.files['image']
    filename = werkzeug.utils.secure_filename(imagefile.filename)
    # frame_paths.append("frames/"+filename)
    frame_paths.append(os.path.join("frames", filename))
    # print("\nReceived image File name : ", frame_paths[-1])
    imagefile.save("frames/" + filename)
    print("\nReceived", len(frame_paths), "frames")
    pred="no"
    if len(frame_paths) == 25:
        pred = run_model()
        frame_paths.clear()
    # if len(frame_paths) > 25:
    #     frame_paths.clear()
    return (pred)

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
@app.route('/showImg', methods=['GET'])
def show_image():
    img = cv2.imread(frame_paths[0])
    cv2.imshow('Cat', img)
    cv2.waitKey(0)
    return "done"

if __name__ == '__main__':
    load_model()  # load model at the beginning once only
    app.run(host='0.0.0.0', port=80)
