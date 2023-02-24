from app import app
# import numpy as np
# import keras.models
# import mediapipe as mp

# def load_model():
#     global model
#     # global frame_paths
#     global frame_rate
#     global mp_holistic
#     global mp_drawing
#     global actions
#     global colors

#     # frame_paths = []
#     frame_rate = 25
#     mp_holistic = mp.solutions.holistic # Holistic model - make our detection
#     mp_drawing = mp.solutions.drawing_utils # Drawing utilities - make our drawings
#     actions = np.array(['NoSign','hello', 'thanks', 'iloveyou'])
#     colors = [(245,117,16), (117,245,16), (16,117,245),(16,117,245)]
#     model = keras.models.load_model('test8.h5')

if __name__ == '__main__':
	app.run()
