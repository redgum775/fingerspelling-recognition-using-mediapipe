import numpy as np
import tensorflow as tf

import pickle

from mediapipe.framework.formats import landmark_pb2
from mediapipe.framework.formats import classification_pb2

from utils import hand_utils

class JaSpellingClassification:
  def __init__(
    self,
    model_path='src/models/ja_fingerspelling_classification/ja_fingerspelling.tflite',
    score_th=0.95,
    invalid_value=0,
    num_threads=1,
  ):
    self.interpreter = tf.lite.Interpreter(model_path=model_path,
                                               num_threads=num_threads)
    self.sc = pickle.load(open("src/models/ja_fingerspelling_classification/scaler.pickle", "rb"))

    self.interpreter.allocate_tensors()
    self.input_details = self.interpreter.get_input_details()
    self.output_details = self.interpreter.get_output_details()

    self.score_th = score_th
    self.invalid_value = invalid_value

    self.TIME_STEPS = 10
    self.explanatory_variables = []

  def __call__(
    self,
    hand_landmarks:landmark_pb2.NormalizedLandmarkList, 
    hand_landmarks_raw:landmark_pb2.LandmarkList, 
    handedness:classification_pb2.ClassificationList
  ):
    self.__update_input_data(hand_landmarks, hand_landmarks_raw, handedness)
    if(len(self.explanatory_variables) == self.TIME_STEPS):
      input_data = np.array([self.sc.transform(self.explanatory_variables)], dtype=np.float32)
      input_details_tensor_index = self.input_details[0]['index']
      self.interpreter.set_tensor(
        input_details_tensor_index,
        input_data)
      self.interpreter.invoke()

      output_details_tensor_index = self.output_details[0]['index']
      result = self.interpreter.get_tensor(output_details_tensor_index)

      result_index = np.argmax(np.squeeze(result))

      if np.squeeze(result)[result_index] < self.score_th:
          result_index = self.invalid_value

      return self.indices_char(result_index)
    else:
      return self.indices_char(0)

  def __update_input_data(
    self,
    hand_landmarks:landmark_pb2.NormalizedLandmarkList, 
    hand_landmarks_raw:landmark_pb2.LandmarkList, 
    handedness:classification_pb2.ClassificationList
  ):
    self.explanatory_variables.append(hand_utils.get_explanatory_variables(hand_landmarks, hand_landmarks_raw, handedness))
    if len(self.explanatory_variables) > self.TIME_STEPS:
      del self.explanatory_variables[0]

  def indices_char(self, index):
    char = ['_',
            'あ','い','う','え','お',
            'か','き','く','け','こ',
            'さ','し','す','せ','そ',
            'た','ち','つ','て','と',
            'な','に','ぬ','ね','の',
            'は','ひ','ふ','へ','ほ',
            'ま','み','む','め','も',
            'や','ゆ','よ',
            'ら','り','る','れ','ろ',
            'わ','を','ん']
    return char[index]