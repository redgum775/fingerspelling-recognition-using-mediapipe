from collections import deque
import cv2

class CvFpsCalc(object):
  def __init__(self, buffer_len=1):
    self._font_size = 0.4
    self._font_width = 1
    self._font_style = cv2.FONT_HERSHEY_COMPLEX
    self._font_color = (255, 255, 255)
    self._background_color = (0, 0, 0)

    self._start_tick = cv2.getTickCount()
    self._freq = 1000.0 / cv2.getTickFrequency()
    self._difftimes = deque(maxlen=buffer_len)
  
  # FPS計算
  def __calc_fps(self):
    current_tick = cv2.getTickCount()
    different_time = (current_tick - self._start_tick) * self._freq
    self._start_tick = current_tick
    self._difftimes.append(different_time)
    fps = 1000.0 / (sum(self._difftimes) / len(self._difftimes))
    fps = round(fps, 2)
    return fps

  # FPSを画面に表示
  def dipsplay_fps(self, image):
    # 文字の背景を黒でオーバーレイ
    height, _= image.shape[:2]
    alpha = 0.75
    overlay = image.copy()
    overlay = cv2.rectangle(overlay, (5, height - 20), (5 + 100, height), self._background_color, -1)
    image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    # FPSの文字入れ
    text = f'FPS: {self.__calc_fps()}'
    cv2.putText(image, text, (5, height - 5), self._font_style, self._font_size, self._font_color, self._font_width)

    return image