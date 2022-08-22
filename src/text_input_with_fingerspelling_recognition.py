import cv2
import mediapipe as mp

import re

from utils import hand_utils
from utils import drawing_utils
from utils import CvFpsCalc
from models import JaSpellingClassification

def main():
  mp_hands = mp.python.solutions.hands

  cvFpsCalc = CvFpsCalc(buffer_len=10)
  finger_spelling_classification = JaSpellingClassification()

  empty_frames = 0
  text = ''

  # For webcam input:
  cap = cv2.VideoCapture(0)
  cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('H', '2', '6', '4'))
  width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
  height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
  fps = cap.get(cv2.CAP_PROP_FPS)
  print(f'INFO: Web camera performance is [width: {width}, height: {height}, fps: {fps}].')

  with mp_hands.Hands(
      max_num_hands = 1,
      min_detection_confidence=0.5,
      min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
      success, image = cap.read()
      if not success:
        print("Ignoring empty camera frame.")
        continue

      image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

      image.flags.writeable = False
      results = hands.process(image)

      image.flags.writeable = True
      image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

      if results.multi_hand_landmarks:
        for id, (hand_landmarks, hand_world_landmarks, handedness) in enumerate(zip(results.multi_hand_landmarks, results.multi_hand_world_landmarks, results.multi_handedness)):
          drawing_utils.draw_bounding_rect(image, hand_landmarks)        
          drawing_utils.draw_landmarks(image, hand_landmarks)
          result_char = finger_spelling_classification(hand_landmarks, hand_world_landmarks, handedness)
          text += result_char
          text = text[-5000:]
          pos = hand_utils.get_bounding_rect_top_left(image, hand_landmarks)
          image = drawing_utils.draw_jp_text(image, matching(text), (pos[0], pos[1]-5))
          print(hand_utils.get_explanatory_variables(hand_landmarks, hand_world_landmarks, handedness))
      else:
        empty_frames += 1
        if(empty_frames==60):
          empty_frames = 0
          text = ''
      if text != '' : print(text)
      image = cvFpsCalc.dipsplay_fps(image)
      cv2.imshow('MediaPipe Hands', image)

      if cv2.waitKey(1) & 0xFF == 27:
        break
  cap.release()

def matching(text):
  return ''.join([c[0] for c in re.split(r'_+', text) if len(c)>=5])

if __name__ == '__main__':
  main()