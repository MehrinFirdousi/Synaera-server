# This is a simple prototype server to receive stream of pictures from the 
# AndroidCamStream client app.
# 
# This server is implemented with the socket.io framework that provides a
# easy-to-use of WebSockets. Therefore, this is a SYNCHRONOUS client/server.
# 
# PLEASE NOTICE: 
# The choice of synch client/server was intended for quick prototyping. However, 
# due to the nature of the video-streaming use-case a more robust implementation
# based on ASYNCHRONOUS client/server communication is strongly recommended. 
# Socket.io also provides an asynch server API.
#
# Created by xetiro (aka Ruben Geraldes) on 28/09/2020.
import sys, getopt
import eventlet
import socketio
import cv2
import numpy as np
from engineio.payload import Payload

import keras.models
import os
import mediapipe as mp
from os.path import join


# Default is 16 which can create a bootleneck for video streaming
Payload.max_decode_packets = 256

sio = socketio.Server()
app = socketio.WSGIApp(sio)

# Default server IP and server Port
ip = "0.0.0.0" 
port = 8080

# Display the image on a OpenCV window
isDisplay = False

# Use authentication to validate users
isAuth = False

# Dummy in-memory key-value pairs user database for dummy authentication using 
# plain-text passwords. Users credentials are username:password key-values
# WARNING: Never use plain-text passwords on a real application.
dummyUserDB = { 
    # Add more users as needed
    "user1": "pass1",
    "user2": "pass2",
    "Alice": "123",
    "Bob": "456"
}

# Map of authenticated session id's and respective usernames
activeSessions = {}
# Map of authenticated usernames and respective session id's
activeUsers = {}
frameCount = [1]


# ------------------------ START SYNAERA ML FUNCTIONS ------------------------

frame_rate = 25
sequence = []
predictions = []
mp_holistic = mp.solutions.holistic # Holistic model - make our detection
mp_drawing = mp.solutions.drawing_utils # Drawing utilities - make our drawings
actions = np.array(['NoSign','hello', 'thanks', 'please', 'sorry', 'you', 'work', 'where'])
# actions = np.array(['NoSign','hello', 'thanks', 'iloveyou'])
model = keras.models.load_model(os.path.join('models', 'Demo.h5'))

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

def run_model(imageBytes):
    result_p = "nothing"
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        nparr = np.frombuffer(imageBytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        # frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        # Make detections
        image, results = mediapipe_detection(frame, holistic)
        
        # Draw landmarks
        draw_landmarks(image, results)

        # Prediction logic
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)

    imgNo = str(len(frameCount))
    cv2.imwrite('frames/img'+imgNo+'.jpg', frame)
    frameCount.append(1)
    last_frames = sequence[-frame_rate:]
    if len(last_frames) == frame_rate:
        res = model.predict(np.expand_dims(last_frames, axis=0))[0]
        print(imgNo, actions[np.argmax(res)])
        if len(predictions) > 0:
            if predictions[-1]==np.argmax(res):
                return "nothing"
        predictions.append(np.argmax(res))
        result_p = actions[np.argmax(res)]
    # if ((int(frame_no) % frame_rate == 0) and (len(last_frames) == frame_rate)):
    # if len(sequence) % 12 == 0 or len(sequence) == 25:
    #     last_frames = sequence[-frame_rate:]
    #     if (len(last_frames) == 25):
    #         res = model.predict(np.expand_dims(last_frames, axis=0))[0]
    #         print(imgNo, actions[np.argmax(res)])
    #         if len(predictions) > 0:
    #             if predictions[-1]==np.argmax(res):
    #                 return "nothing"
    #         predictions.append(np.argmax(res))
    #         result_p = actions[np.argmax(res)]
    return (result_p)

# ------------------------ END SYNAERA ML FUNCTIONS ------------------------

@sio.event
def connect(sid, environ):
    print('connect', sid)
    print(activeUsers)

# Method used for user "dummy" authentication using an in-memory dummy database. 
# This can be used to authenticate the user with other server/service.
# WARNING: never use plain-text passwords on a real application.
@sio.event
def authenticate(sid, username, password, clientCallbackEvent):
    user = dummyUserDB.get(username)
    if isAuth == False or (user is not None and user == password):
        # add username to the session
        addUserSession(sid, username)
        sio.emit(clientCallbackEvent, True)
        print("User [" + username +"] authenticated.")
    else:
        sio.emit(clientCallbackEvent, False)
        sio.sleep(2) # give time to the socket emit the callback to the user
        sio.disconnect(sid)
        print("User [" + username +"] authentication failed.")

# This is the main method that the client calls when streaming the pictures to 
# the server. Each receiveImage event is already processed in a new thread.
# The image format is JPEG and is sent by the client in as binary data of byte[] 
# received in python as Bytes.
@sio.event
def receiveImage(sid, imageBytes):
    # HINT: Process the image here or send image to another server here
    run_model(imageBytes)
    if(isDisplay):
        displayImage(activeSessions[sid], bytes(imageBytes))

@sio.event
def disconnect(sid):
    print('disconnect', sid)
    deleteUserSession(sid)
    print(activeUsers)
    if isDisplay:
        cv2.destroyAllWindows() # Doesn't work well in Unix environments = Zombie window
        cv2.waitKey(1)

def addUserSession(sid, username):
    activeSessions[sid] = username
    activeUsers[username] = sid

def deleteUserSession(sid):
    username = activeSessions[sid]
    activeSessions.pop(sid, None)
    activeUsers.pop(username, None)

def displayImage(username, imageBytes):
    # Decode image from bytes
    nparr = np.frombuffer(imageBytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    imgNo = str(len(frameCount))
    cv2.imwrite('frames/img'+imgNo+'.jpg', img)
    frameCount.append(1)
    print("received ", imgNo)
    # Show image after decoded
    # cv2.namedWindow(username, cv2.WINDOW_AUTOSIZE)
    # cv2.imshow(username, img)
    # cv2.waitKey(1)

def executeCommandArgs(argv):
    global ip, port, isDisplay, isAuth, frameCount
    scriptName = argv[0]
    try:
        opts, args = getopt.getopt(argv[1:], "adhi:p:", ["ip=", "port=", "display", "auth"])
    except getopt.GetoptError: # wrong commands
        print(scriptName + " -i <server_ip> -p <server_port>")
        sys.exit(2)

    for opt, arg in opts:
        if opt == "-h": # help command
            print(scriptName + " -i <server_ip> -p <server_port")
            sys.exit()
        elif opt in ("-i", "--ip"):
            ip = arg
        elif opt in ("-p", "--port"):
            port = int(arg)
        elif opt in ("-d", "--display"):
            isDisplay = True
        elif opt in ("-a", "--auth"):
            isAuth = True

if __name__ == '__main__':
    executeCommandArgs(sys.argv)
    eventlet.wsgi.server(eventlet.listen((ip, port)), app)