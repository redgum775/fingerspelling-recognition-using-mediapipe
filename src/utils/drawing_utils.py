import cv2
import numpy as np
import platform

from PIL import Image, ImageDraw, ImageFont

from utils import hand_utils
from mediapipe.framework.formats import landmark_pb2

# setting the font use.
SYSTEM = platform.system()
if SYSTEM == 'Linux':
  _FONT = '/usr/share/fonts/opentype/noto/NotoSerifCJK-Light.ttc'
elif SYSTEM == 'Windows':
  _FONT = 'C:/Windows/Fonts/msgothic.ttc'
elif SYSTEM == 'Darwin':
  _FONT = '/System/Library/Fonts/ヒラギノ角ゴシック W4.ttc'

_BLACK = (0, 0, 0)
_WHITE = (255, 255, 255)

# 手の描画
def draw_landmarks(image, hand_landmarks:landmark_pb2.LandmarkList):
  height, width = image.shape[:2]

  # 掌
  cv2.line(image, (int(hand_landmarks.landmark[0].x * width), int(hand_landmarks.landmark[0].y * height)),
                  (int(hand_landmarks.landmark[1].x * width), int(hand_landmarks.landmark[1].y * height)),
                  _BLACK, 6)
  cv2.line(image, (int(hand_landmarks.landmark[0].x * width), int(hand_landmarks.landmark[0].y * height)),
                  (int(hand_landmarks.landmark[1].x * width), int(hand_landmarks.landmark[1].y * height)),
                  _WHITE, 2)
  cv2.line(image, (int(hand_landmarks.landmark[0].x * width), int(hand_landmarks.landmark[0].y * height)),
                  (int(hand_landmarks.landmark[5].x * width), int(hand_landmarks.landmark[5].y * height)),
                  _BLACK, 6)
  cv2.line(image, (int(hand_landmarks.landmark[0].x * width), int(hand_landmarks.landmark[0].y * height)),
                  (int(hand_landmarks.landmark[5].x * width), int(hand_landmarks.landmark[5].y * height)),
                  _WHITE, 2)
  cv2.line(image, (int(hand_landmarks.landmark[5].x * width), int(hand_landmarks.landmark[5].y * height)),
                  (int(hand_landmarks.landmark[9].x * width), int(hand_landmarks.landmark[9].y * height)),
                  _BLACK, 6)
  cv2.line(image, (int(hand_landmarks.landmark[5].x * width), int(hand_landmarks.landmark[5].y * height)),
                  (int(hand_landmarks.landmark[9].x * width), int(hand_landmarks.landmark[9].y * height)),
                  _WHITE, 2)
  cv2.line(image, (int(hand_landmarks.landmark[9].x * width), int(hand_landmarks.landmark[9].y * height)),
                  (int(hand_landmarks.landmark[13].x * width), int(hand_landmarks.landmark[13].y * height)),
                  _BLACK, 6)
  cv2.line(image, (int(hand_landmarks.landmark[9].x * width), int(hand_landmarks.landmark[9].y * height)),
                  (int(hand_landmarks.landmark[13].x * width), int(hand_landmarks.landmark[13].y * height)),
                  _WHITE, 2)
  cv2.line(image, (int(hand_landmarks.landmark[13].x * width), int(hand_landmarks.landmark[13].y * height)),
                  (int(hand_landmarks.landmark[17].x * width), int(hand_landmarks.landmark[17].y * height)),
                  _BLACK, 6)
  cv2.line(image, (int(hand_landmarks.landmark[13].x * width), int(hand_landmarks.landmark[13].y * height)),
                  (int(hand_landmarks.landmark[17].x * width), int(hand_landmarks.landmark[17].y * height)),
                  _WHITE, 2)                  
  cv2.line(image, (int(hand_landmarks.landmark[0].x * width), int(hand_landmarks.landmark[0].y * height)),
                  (int(hand_landmarks.landmark[17].x * width), int(hand_landmarks.landmark[17].y * height)),
                  _BLACK, 6)
  cv2.line(image, (int(hand_landmarks.landmark[0].x * width), int(hand_landmarks.landmark[0].y * height)),
                  (int(hand_landmarks.landmark[17].x * width), int(hand_landmarks.landmark[17].y * height)),
                  _WHITE, 2)

  # 親指
  cv2.line(image, (int(hand_landmarks.landmark[1].x * width), int(hand_landmarks.landmark[1].y * height)),
                  (int(hand_landmarks.landmark[2].x * width), int(hand_landmarks.landmark[2].y * height)),
                  _BLACK, 6)
  cv2.line(image, (int(hand_landmarks.landmark[1].x * width), int(hand_landmarks.landmark[1].y * height)),
                  (int(hand_landmarks.landmark[2].x * width), int(hand_landmarks.landmark[2].y * height)),
                  _WHITE, 2)
  cv2.line(image, (int(hand_landmarks.landmark[2].x * width), int(hand_landmarks.landmark[2].y * height)),
                  (int(hand_landmarks.landmark[3].x * width), int(hand_landmarks.landmark[3].y * height)),
                  _BLACK, 6)
  cv2.line(image, (int(hand_landmarks.landmark[2].x * width), int(hand_landmarks.landmark[2].y * height)),
                  (int(hand_landmarks.landmark[3].x * width), int(hand_landmarks.landmark[3].y * height)),
                  _WHITE, 2)
  cv2.line(image, (int(hand_landmarks.landmark[3].x * width), int(hand_landmarks.landmark[3].y * height)),
                  (int(hand_landmarks.landmark[4].x * width), int(hand_landmarks.landmark[4].y * height)),
                  _BLACK, 6)
  cv2.line(image, (int(hand_landmarks.landmark[3].x * width), int(hand_landmarks.landmark[3].y * height)),
                  (int(hand_landmarks.landmark[4].x * width), int(hand_landmarks.landmark[4].y * height)),
                  _WHITE, 2)

  # 人指し指
  cv2.line(image, (int(hand_landmarks.landmark[5].x * width), int(hand_landmarks.landmark[5].y * height)),
                  (int(hand_landmarks.landmark[6].x * width), int(hand_landmarks.landmark[6].y * height)),
                  _BLACK, 6)
  cv2.line(image, (int(hand_landmarks.landmark[5].x * width), int(hand_landmarks.landmark[5].y * height)),
                  (int(hand_landmarks.landmark[6].x * width), int(hand_landmarks.landmark[6].y * height)),
                  _WHITE, 2)
  cv2.line(image, (int(hand_landmarks.landmark[6].x * width), int(hand_landmarks.landmark[6].y * height)),
                  (int(hand_landmarks.landmark[7].x * width), int(hand_landmarks.landmark[7].y * height)),
                  _BLACK, 6)
  cv2.line(image, (int(hand_landmarks.landmark[6].x * width), int(hand_landmarks.landmark[6].y * height)),
                  (int(hand_landmarks.landmark[7].x * width), int(hand_landmarks.landmark[7].y * height)),
                  _WHITE, 2)
  cv2.line(image, (int(hand_landmarks.landmark[7].x * width), int(hand_landmarks.landmark[7].y * height)),
                  (int(hand_landmarks.landmark[8].x * width), int(hand_landmarks.landmark[8].y * height)),
                  _BLACK, 6)
  cv2.line(image, (int(hand_landmarks.landmark[7].x * width), int(hand_landmarks.landmark[7].y * height)),
                  (int(hand_landmarks.landmark[8].x * width), int(hand_landmarks.landmark[8].y * height)),
                  _WHITE, 2)

  # 中指
  cv2.line(image, (int(hand_landmarks.landmark[9].x * width), int(hand_landmarks.landmark[9].y * height)),
                  (int(hand_landmarks.landmark[10].x * width), int(hand_landmarks.landmark[10].y * height)),
                  _BLACK, 6)
  cv2.line(image, (int(hand_landmarks.landmark[9].x * width), int(hand_landmarks.landmark[9].y * height)),
                  (int(hand_landmarks.landmark[10].x * width), int(hand_landmarks.landmark[10].y * height)),
                  _WHITE, 2)
  cv2.line(image, (int(hand_landmarks.landmark[10].x * width), int(hand_landmarks.landmark[10].y * height)),
                  (int(hand_landmarks.landmark[11].x * width), int(hand_landmarks.landmark[11].y * height)),
                  _BLACK, 6)
  cv2.line(image, (int(hand_landmarks.landmark[10].x * width), int(hand_landmarks.landmark[10].y * height)),
                  (int(hand_landmarks.landmark[11].x * width), int(hand_landmarks.landmark[11].y * height)),
                  _WHITE, 2)
  cv2.line(image, (int(hand_landmarks.landmark[11].x * width), int(hand_landmarks.landmark[11].y * height)),
                  (int(hand_landmarks.landmark[12].x * width), int(hand_landmarks.landmark[12].y * height)),
                  _BLACK, 6)
  cv2.line(image, (int(hand_landmarks.landmark[11].x * width), int(hand_landmarks.landmark[11].y * height)),
                  (int(hand_landmarks.landmark[12].x * width), int(hand_landmarks.landmark[12].y * height)),
                  _WHITE, 2)

  # 薬指
  cv2.line(image, (int(hand_landmarks.landmark[13].x * width), int(hand_landmarks.landmark[13].y * height)),
                  (int(hand_landmarks.landmark[14].x * width), int(hand_landmarks.landmark[14].y * height)),
                  _BLACK, 6)
  cv2.line(image, (int(hand_landmarks.landmark[13].x * width), int(hand_landmarks.landmark[13].y * height)),
                  (int(hand_landmarks.landmark[14].x * width), int(hand_landmarks.landmark[14].y * height)),
                  _WHITE, 2)
  cv2.line(image, (int(hand_landmarks.landmark[14].x * width), int(hand_landmarks.landmark[14].y * height)),
                  (int(hand_landmarks.landmark[15].x * width), int(hand_landmarks.landmark[15].y * height)),
                  _BLACK, 6)
  cv2.line(image, (int(hand_landmarks.landmark[14].x * width), int(hand_landmarks.landmark[14].y * height)),
                  (int(hand_landmarks.landmark[15].x * width), int(hand_landmarks.landmark[15].y * height)),
                  _WHITE, 2)
  cv2.line(image, (int(hand_landmarks.landmark[15].x * width), int(hand_landmarks.landmark[15].y * height)),
                  (int(hand_landmarks.landmark[16].x * width), int(hand_landmarks.landmark[16].y * height)),
                  _BLACK, 6)
  cv2.line(image, (int(hand_landmarks.landmark[15].x * width), int(hand_landmarks.landmark[15].y * height)),
                  (int(hand_landmarks.landmark[16].x * width), int(hand_landmarks.landmark[16].y * height)),
                  _WHITE, 2)

  # 小指
  cv2.line(image, (int(hand_landmarks.landmark[17].x * width), int(hand_landmarks.landmark[17].y * height)),
                  (int(hand_landmarks.landmark[18].x * width), int(hand_landmarks.landmark[18].y * height)),
                  _BLACK, 6)
  cv2.line(image, (int(hand_landmarks.landmark[17].x * width), int(hand_landmarks.landmark[17].y * height)),
                  (int(hand_landmarks.landmark[18].x * width), int(hand_landmarks.landmark[18].y * height)),
                  _WHITE, 2)
  cv2.line(image, (int(hand_landmarks.landmark[18].x * width), int(hand_landmarks.landmark[18].y * height)),
                  (int(hand_landmarks.landmark[19].x * width), int(hand_landmarks.landmark[19].y * height)),
                  _BLACK, 6)
  cv2.line(image, (int(hand_landmarks.landmark[18].x * width), int(hand_landmarks.landmark[18].y * height)),
                  (int(hand_landmarks.landmark[19].x * width), int(hand_landmarks.landmark[19].y * height)),
                  _WHITE, 2)
  cv2.line(image, (int(hand_landmarks.landmark[19].x * width), int(hand_landmarks.landmark[19].y * height)),
                  (int(hand_landmarks.landmark[20].x * width), int(hand_landmarks.landmark[20].y * height)),
                  _BLACK, 6)
  cv2.line(image, (int(hand_landmarks.landmark[19].x * width), int(hand_landmarks.landmark[19].y * height)),
                  (int(hand_landmarks.landmark[20].x * width), int(hand_landmarks.landmark[20].y * height)),
                  _WHITE, 2)

  # ランドマークポイント
  for landmark in hand_landmarks.landmark:
    cv2.circle(image, (int(landmark.x * width), int(landmark.y * height)), 5, _WHITE, -1)
    cv2.circle(image, (int(landmark.x * width), int(landmark.y * height)), 5, _BLACK, 1)

  return image

def draw_bounding_rect(image, hand_landmarks:landmark_pb2.LandmarkList):
  pt = hand_utils.calc_bounding_rect(image, hand_landmarks)
  cv2.rectangle(image, (pt[0], pt[1]), (pt[2],pt[3]), _BLACK, 2)
  return image

def __pil2cv(imgPIL):
  imgCV = np.array(imgPIL)
  return imgCV

def __cv2pil(imgCV):
  imgPIL = Image.fromarray(imgCV)
  return imgPIL

def __cv2_putText(image, text, pos, fontFace, fontScale, color):
  x, y = pos
  imgPIL = __cv2pil(image)
  draw = ImageDraw.Draw(imgPIL)
  fontPIL = ImageFont.truetype(font=fontFace, size=fontScale)
  _, h = draw.textsize(text, font=fontPIL)
  draw.text(xy=(x,y-h), text=text, fill=color, font=fontPIL)
  imgCV = __pil2cv(imgPIL)
  return imgCV

def draw_jp_text(image, text, pos, scale=25, color=_WHITE):
  overlay = image.copy()
  x,y = pos
  overlay = cv2.rectangle(overlay, (x,y-scale), (x+len(text)*scale,y), (0, 0, 0), -1)
  image = overlay_image(image, overlay)
  image = __cv2_putText(image, text, pos, _FONT, scale, color)
  return image
  
def overlay_image(image, overlay, alpha=0.75):
  image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
  return image