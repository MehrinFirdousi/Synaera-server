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

import re
from keras.layers import Input, LSTM, Embedding, Dense
from keras.models import Model
import time
import ast

model_weights='Model_10ws_4p.h5'
# Default is 16 which can create a bootleneck for video streaming
Payload.max_decode_packets = 256

sio = socketio.Server()
app = socketio.WSGIApp(sio)

# Default server IP and server Port
ip = "0.0.0.0" 
port = 5000

# Display the image on a OpenCV window
isDisplay = False

# Use authentication to validate users
isAuth = False

# Dummy in-memory key-value pairs user database for dummy authentication using 
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

class colors:
    RED_BOLD = '\033[91m' + '\033[1m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    UNDERLINE = '\033[4m'
    UNDERLINE_GREEN = '\033[4m' + '\033[92m'

# ------------------------ START SYNAERA CV FUNCTIONS ------------------------
# ------------------------ START SYNAERA CV FUNCTIONS ------------------------
# ------------------------ START SYNAERA CV FUNCTIONS ------------------------
# ------------------------ START SYNAERA CV FUNCTIONS ------------------------
# ------------------------ START SYNAERA CV FUNCTIONS ------------------------
# ------------------------ START SYNAERA CV FUNCTIONS ------------------------
# ------------------------ START SYNAERA CV FUNCTIONS ------------------------

frame_rate = 25
frames = []
sequence = []
predictions = []
mp_holistic = mp.solutions.holistic # Holistic model - make our detection
mp_drawing = mp.solutions.drawing_utils # Drawing utilities - make our drawings
# actions = np.array(['NoSign','hello', 'thanks', 'please', 'sorry', 'you', 'work', 'where'])
actions = np.array([['NoSign','hello','thanks','sorry','you','work','where']])
# actions = np.array(['NoSign','hello', 'thanks', 'iloveyou'])
cv_model = keras.models.load_model(os.path.join('models', model_weights))

# To extract keypoint values from frame using mediapipe
def mediapipe_detection(image, cv_model):
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
	image.flags.writeable = False                  # Image is no longer writeable
	results = cv_model.process(image)                 # Make prediction
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

def run_model_frame_batches(imageBytes):
	sequence = []
	result_p = "nothing"
	nparr = np.frombuffer(imageBytes, np.uint8)
	frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
	frames.append(frame)
	# cv2.imwrite('frames/img'+str(len(frames))+'.jpg', frame)
	print("received", len(frames))
	if (len(frames) == frame_rate):
		with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
			for frame in frames:
				# Make detections
				image, results = mediapipe_detection(frame, holistic)
				
				# Draw landmarks
				draw_landmarks(image, results)

				# Prediction logic
				keypoints = extract_keypoints(results)
				sequence.append(keypoints)
			frames.clear()
			res = cv_model.predict(np.expand_dims(sequence, axis=0))[0]
			print(actions[np.argmax(res)])
			# check if prediction is nosign and predictions array is empty
			if len(predictions) == 0:
				if np.argmax(res) == 0:
					return "nothing"
			# check duplicate prediction
			if len(predictions) > 0:
				if predictions[-1]==np.argmax(res):
					return "nothing"

			predictions.append(np.argmax(res))
			result_p = actions[np.argmax(res)]
	return (result_p)

def run_model(imageBytes):
	result_p = "nothing"
	with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
		nparr = np.frombuffer(imageBytes, np.uint8)
		frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
		# frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
		image, results = mediapipe_detection(frame, holistic)
		draw_landmarks(image, results)
		keypoints = extract_keypoints(results)
		sequence.append(keypoints)

	imgNo = str(len(frameCount))
	# cv2.imwrite('frames/img'+imgNo+'.jpg', frame)
	frameCount.append(1)
	last_frames = sequence[-frame_rate:]
	if len(last_frames) == frame_rate:
		res = cv_model.predict(np.expand_dims(last_frames, axis=0))[0]
		print(imgNo, actions[np.argmax(res)])
		if len(predictions) > 0:
			if predictions[-1]==np.argmax(res):
				return "nothing"
		predictions.append(np.argmax(res))
		result_p = actions[np.argmax(res)]
	
	return (result_p)

# ------------------------ END SYNAERA CV FUNCTIONS ------------------------
# ------------------------ END SYNAERA CV FUNCTIONS ------------------------
# ------------------------ END SYNAERA CV FUNCTIONS ------------------------
# ------------------------ END SYNAERA CV FUNCTIONS ------------------------
# ------------------------ END SYNAERA CV FUNCTIONS ------------------------
# ------------------------ END SYNAERA CV FUNCTIONS ------------------------
# ------------------------ END SYNAERA CV FUNCTIONS ------------------------

# ------------------------ START SYNAERA NLP FUNCTIONS ------------------------
# ------------------------ START SYNAERA NLP FUNCTIONS ------------------------
# ------------------------ START SYNAERA NLP FUNCTIONS ------------------------
# ------------------------ START SYNAERA NLP FUNCTIONS ------------------------
# ------------------------ START SYNAERA NLP FUNCTIONS ------------------------
# ------------------------ START SYNAERA NLP FUNCTIONS ------------------------
# ------------------------ START SYNAERA NLP FUNCTIONS ------------------------
# ------------------------ START SYNAERA NLP FUNCTIONS ------------------------
# ------------------------ START SYNAERA NLP FUNCTIONS ------------------------
# ------------------------ START SYNAERA NLP FUNCTIONS ------------------------
# ------------------------ START SYNAERA NLP FUNCTIONS ------------------------
# ------------------------ START SYNAERA NLP FUNCTIONS ------------------------

def read_list_from_file():
	inputFile = open( "myVars.txt", "r")
	lines = inputFile.readlines()

	objects = []
	for line in lines:
		objects.append(ast.literal_eval(line))
		
	return objects[0][0], objects[0][1], objects[0][2], objects[0][3], objects[0][4], objects[0][5], objects[0][6]

# get the start time
st_final = time.time()
st = time.time()

max_length_src, max_length_tar, num_encoder_tokens, num_decoder_tokens, input_token_index, target_token_index, reverse_target_char_index = read_list_from_file()

print(colors.UNDERLINE_GREEN + 'Importing Variables:' + colors.ENDC, round(time.time() - st, 2), 'seconds')
st = time.time()

latent_dim = 50

# Encoder
encoder_inputs = Input(shape=(None,))
enc_emb =  Embedding(num_encoder_tokens, latent_dim, mask_zero = True)(encoder_inputs)
encoder_lstm = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(enc_emb)
# We discard `encoder_outputs` and only keep the states.
encoder_states = [state_h, state_c]

# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = Input(shape=(None,))
dec_emb_layer = Embedding(num_decoder_tokens, latent_dim, mask_zero = True)
dec_emb = dec_emb_layer(decoder_inputs)

'''
We set up our decoder to return full output sequences, and to return internal states as well. 
We don't use the return states in the training model, but we will use them in inference.
'''
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(dec_emb, initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)
# Define the model that will turn `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
nlp_model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

print(colors.UNDERLINE_GREEN + 'Setting up Model:' + colors.ENDC, round(time.time() - st, 2), 'seconds')
st = time.time()

nlp_model.load_weights('nmt_weights_v5.h5')

print(colors.UNDERLINE_GREEN + 'Loading Weights:' + colors.ENDC, round(time.time() - st, 2), 'seconds')
st = time.time()

### INFERENCING ###
encoder_model = Model(encoder_inputs, encoder_states) # Encode the input sequence to get the "thought vectors"

# Decoder setup - Below tensors will hold the states of the previous time step
decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
dec_emb2 = dec_emb_layer(decoder_inputs) # Get the embeddings of the decoder sequence

# To predict the next word in the sequence, set the initial states to the states from the previous time step
decoder_outputs2, state_h2, state_c2 = decoder_lstm(dec_emb2, initial_state=decoder_states_inputs)
decoder_states2 = [state_h2, state_c2]
decoder_outputs2 = decoder_dense(decoder_outputs2) # A dense softmax layer to generate prob dist. over the target vocabulary

# Final decoder model
decoder_model = Model(
	[decoder_inputs] + decoder_states_inputs,
	[decoder_outputs2] + decoder_states2)

print(colors.UNDERLINE_GREEN + 'Setting up Decoder:' + colors.ENDC, round(time.time() - st, 2), 'seconds')
st = time.time()

# Reverse-lookup token index to decode sequences back to something readable.
def decode_sequence(input_text):
	encoder_input_data = np.zeros((1, max_length_src), dtype='float32')
	error_word = ''
	try:
		for i, input_text in enumerate([input_text]):
			#print(colors.WARNING + "i:", i, " | input_text: ", input_text, "" + colors.ENDC)
			for t, word in enumerate(input_text.split()):
				error_word = word
				encoder_input_data[i, t] = input_token_index[word]
	except:
		return colors.RED_BOLD + '"' + error_word + '" doesn\'t exist in the dataset.' + colors.ENDC
		
	states_value = encoder_model.predict(encoder_input_data)
		
	target_seq = np.zeros((1, 1))
	target_seq[0, 0] = target_token_index['START_']
	stop_condition = False
	decoded_sentence = ''

	while not stop_condition:
		output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
		sampled_token_index = np.argmax(output_tokens[0, -1, :])
		sampled_char = reverse_target_char_index[sampled_token_index]
		decoded_sentence += ' ' + sampled_char
		
		if (sampled_char == '_END' or len(decoded_sentence) > 50):
			stop_condition = True
		
		target_seq = np.zeros((1, 1))
		target_seq[0, 0] = sampled_token_index
		states_value = [h, c]
		
	return decoded_sentence[:-4]

def preprocess_sentence(sentence):
    # lower case to standardize the sentence and remove extra spaces
    sentence = sentence.lower().strip()
    # if QM-wig or 6 Ws or How is in the sentence, then it is a question
    words = ['who', 'what', 'when', 'where', 'why', 'how']
    question_flag = 0
    if 'qm-wig' in sentence or any(word in sentence for word in words):
        question_flag = 1
    sentence = sentence.replace('qm-wig', '')

    # remove punctuation (isn't required but im still including it)
    sentence = re.sub(r"([?.!,])", "", sentence)
    # replace numbers with words
    number_replacements = {'1': " one ", '2':" two ", '3':" three ", '4':" four ", 
                           '5':" five ", '6':" six ", '7':" seven ", '8':" eight ", 
                           '9':" nine ", '0':" zero "}
    for key, value in number_replacements.items():
        sentence = sentence.replace(key, value)
    # remove extra spaces
    sentence = re.sub(r'[" "]+', " ", sentence)
    sentence = sentence.strip()

    words = sentence.split()
    result = []
    # Empty temporary list to store single letters
    temp = []
    for word in words:
        if len(word) == 1:
            temp.append(word)
        else:
            # If there are any single letters in the temporary list,
            # join them with a dash and append to the result list
            if temp:
                result.append('-'.join(temp))
                temp = []
            # Append the non-single letter word to the result list
            result.append(word)
    if temp:
        result.append('-'.join(temp))
    
    # Save the dashed words in a list so that it can be replaced later
    replaced_words = [match for match in result if "-" in match]
    # Replace the single letters with 'XXXXX' in the result list
    result = ["xxxxx" if '-' in element else element for element in result]
    # Join the words in the result list back into a string sentence
    sentence = ' '.join(result)

    return sentence, question_flag, replaced_words

def gloss_to_english():
	glossInput = ""
	decoded_sentence = ""
	# if len(predictions) == 1 and predictions[-1] == 0:
	# 	predictions.clear()
	if len(predictions) > 1:
		# last sign was nosign
		if predictions[-1] == 0:
			for res in predictions[:-1]:
				glossInput += actions[res] + " "
			glossInput = glossInput[:-1]
			print(glossInput)
			prep_input, question_flag, replaced_words = preprocess_sentence(glossInput)
			# if only 1 word is given, then no need to decode
			decoded_sentence = decode_sequence(prep_input) if len(prep_input.split()) > 1 else prep_input

			# if '?' not in decoded sentence and original input had 'QM-wig' then add '?' at the end
			if '?' not in decoded_sentence and question_flag == 1:
				decoded_sentence = decoded_sentence.strip() + '?'

			# Replace the 'XXXXX' with the original single letter words
			for word in replaced_words:
				decoded_sentence = decoded_sentence.replace('xxxxx', word.replace('-',''), 1)
			decoded_sentence = decoded_sentence.replace('xxxxx', '')
			
			# if decoded sentence contains ['who', 'what', 'when', 'where', 'why', 'how'] then add '?' at the end
			if any(word in decoded_sentence for word in ['who', 'what', 'when', 'where', 'why', 'how']) and '?' not in decoded_sentence:
				decoded_sentence = decoded_sentence.strip() + '?'
			print("decoded:", decoded_sentence)
			predictions.clear()
	return (decoded_sentence)
	

# ------------------------ END SYNAERA NLP FUNCTIONS ------------------------
# ------------------------ END SYNAERA NLP FUNCTIONS ------------------------
# ------------------------ END SYNAERA NLP FUNCTIONS ------------------------
# ------------------------ END SYNAERA NLP FUNCTIONS ------------------------
# ------------------------ END SYNAERA NLP FUNCTIONS ------------------------
# ------------------------ END SYNAERA NLP FUNCTIONS ------------------------
# ------------------------ END SYNAERA NLP FUNCTIONS ------------------------
# ------------------------ END SYNAERA NLP FUNCTIONS ------------------------
# ------------------------ END SYNAERA NLP FUNCTIONS ------------------------
# ------------------------ END SYNAERA NLP FUNCTIONS ------------------------
# ------------------------ END SYNAERA NLP FUNCTIONS ------------------------


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
def receiveImage(sid, imageBytes, clientCallBackEvent):
	# HINT: Process the image here or send image to another server here
	gloss = run_model_frame_batches(imageBytes)
	real_text = gloss_to_english()
	if gloss != "nothing":
		if (len(real_text) == 0):
			sio.emit(clientCallBackEvent, gloss)
			print("gloss result:", gloss)
		else:
			sio.emit(clientCallBackEvent, real_text)
			print("real result:", real_text)
	# if(isDisplay):
	# 	displayImage(activeSessions[sid], bytes(imageBytes))

@sio.event
def disconnect(sid):
	print('disconnect', sid)
	# sequence.clear()
	# predictions.clear()
	frameCount.clear()
	frames.clear()
	print("cleared prediction data")
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
	# cv2.imwrite('frames/img'+imgNo+'.jpg', img)
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