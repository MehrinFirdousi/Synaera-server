import keras.models
import os
import mediapipe as mp
import numpy as np
import cv2
from os.path import join

model_weights='Model_10ws_4p.h5'
frame_rate = 25
frames = []
sequence = []
predictions = []
mp_holistic = mp.solutions.holistic # Holistic model - make our detection
mp_drawing = mp.solutions.drawing_utils # Drawing utilities - make our drawings
# actions = np.array(['NoSign','hello', 'thanks', 'please', 'sorry', 'you', 'work', 'where'])
# actions = np.array(['NoSign','hello', 'thanks', 'iloveyou'])
actions = np.array(['NoSign', 'hello', 'you', 'work', 'where', 'how', 'your', 'day', 'b', 'o'])
cv_wts = keras.models.load_model(os.path.join('models', model_weights))
frameCount = [1]

# To extract keypoint values from frame using mediapipe
def mediapipe_detection(image, cv_wts):
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
	image.flags.writeable = False                  # Image is no longer writeable
	results = cv_wts.process(image)                 # Make prediction
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
	frame = cv2.flip(frame, 1)
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
			res = cv_wts.predict(np.expand_dims(sequence, axis=0))[0]
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
		res = cv_wts.predict(np.expand_dims(last_frames, axis=0))[0]
		print(imgNo, actions[np.argmax(res)])
		if len(predictions) > 0:
			if predictions[-1]==np.argmax(res):
				return "nothing"
		predictions.append(np.argmax(res))
		result_p = actions[np.argmax(res)]
	
	return (result_p)