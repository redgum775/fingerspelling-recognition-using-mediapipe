import numpy as np
import cv2

from mediapipe.framework.formats import landmark_pb2
from mediapipe.framework.formats import classification_pb2

# 3点間の角度を計算 (2次元)(0~360°)
def __calc_2D_angle(a:landmark_pb2.Landmark, b:landmark_pb2.Landmark, j:landmark_pb2.Landmark=None):
  j_array = np.array([j.x, j.y]) if j is not None else np.array([0,0])
  a_array = np.array([a.x, a.y])
  b_array = np.array([b.x, b.y])
  vec = (a_array - j_array) - (b_array - j_array)
  rad =  np.arctan2(vec[1], vec[0])
  theta = np.degrees(rad) + 180
  return theta

# 3点間の角度を計算 (3次元)(0~180°)
def __calc_3D_angle(a:landmark_pb2.Landmark, b:landmark_pb2.Landmark, j:landmark_pb2.Landmark=None):
  j_array = np.array([j.x, j.y, j.z]) if j is not None else np.array([0,0,0])
  a_array = np.array([a.x, a.y, a.z])
  b_array = np.array([b.x, b.y, b.z])
  vec_a = a_array - j_array
  vec_b = b_array - j_array
  dot_ab = np.dot(vec_a, vec_b)
  norm_a = np.linalg.norm(vec_a)
  norm_b = np.linalg.norm(vec_b)
  cos = dot_ab / (norm_a * norm_b)
  rad = np.arccos(cos)
  theta = np.degrees(rad)
  return theta

# 手の角度のリストを返す(15個)
def calc_joint_angles(hand_world_landmarks:landmark_pb2.LandmarkList):
  angles = []
  for idx, _ in enumerate(hand_world_landmarks.landmark):
    if 1 <= idx and idx <= 20:
      if (idx % 4 != 0) and ((idx - 1) % 4 != 0):
        angle = __calc_3D_angle(hand_world_landmarks.landmark[idx-1], hand_world_landmarks.landmark[idx+1], hand_world_landmarks.landmark[idx])
        angles.append(angle)
      elif ((idx - 1) % 4) == 0:
        angle = __calc_3D_angle(hand_world_landmarks.landmark[0], hand_world_landmarks.landmark[idx+1], hand_world_landmarks.landmark[idx])
        angles.append(angle)
  return angles

# 手の向き(上下左右を度数法)
def calc_direction_angle(hand_landmarks:landmark_pb2.NormalizedLandmarkList):
  return __calc_2D_angle(hand_landmarks.landmark[0], hand_landmarks.landmark[9])

# 掌の向き(正面 or 後ろ)
def is_plam_facing(
  hand_landmarks:landmark_pb2.NormalizedLandmarkList, 
  handedness:classification_pb2.ClassificationList
):
  top = -1 if 45 <= __calc_2D_angle(hand_landmarks.landmark[0], hand_landmarks.landmark[9]) <= 125 else 1
  side = 1 if handedness.classification[0].label == 'Right' else -1
  if (hand_landmarks.landmark[5].x * side * top) < (hand_landmarks.landmark[17].x * side * top):
    return 1
  else:
    return 0

# 2点間の距離を計算
def calc_distance(a:landmark_pb2.Landmark, b:landmark_pb2.Landmark):
  a_array = np.array([a.x, a.y, a.z])
  b_array = np.array([b.x, b.y, b.z])
  return np.linalg.norm(a_array - b_array)

# 親指と人差し指間の距離
def calc_distance_to_thumb_and_index_finger(hand_world_landmarks:landmark_pb2.LandmarkList):
  return calc_distance(hand_world_landmarks.landmark[4], hand_world_landmarks.landmark[8])

# 親指と中指間の距離
def calc_distance_to_thumb_and_middle_finger(hand_world_landmarks:landmark_pb2.LandmarkList):
  return calc_distance(hand_world_landmarks.landmark[4], hand_world_landmarks.landmark[12])

# 人差し指と中指間の距離
def calc_distance_to_index_and_middle_finger(hand_world_landmarks:landmark_pb2.LandmarkList):
  return calc_distance(hand_world_landmarks.landmark[8], hand_world_landmarks.landmark[12])

def calc_distance_to_hand_and_hand(hand_landmarks_1:landmark_pb2.NormalizedLandmarkList, hand_landmarks_2:landmark_pb2.NormalizedLandmarkList):
  sam = 0
  for hand_landmark_1, hand_landmark2 in hand_landmarks_1.landmark, hand_landmarks_2.landmark:
    sam += calc_distance(hand_landmark_1, hand_landmark2)
  avg = sam / 21
  return avg

# a1-a2, b1-b2の2線分が交差しているかを確かめる
def __is_intersect(
  a1:landmark_pb2.Landmark, 
  a2:landmark_pb2.Landmark, 
  b1:landmark_pb2.Landmark, 
  b2:landmark_pb2.Landmark
):
  a1_array = np.array([a1.x, a1.y])
  a2_array = np.array([a2.x, a2.y])
  b1_array = np.array([b1.x, b1.y])
  b2_array = np.array([b2.x, b2.y])
  t1 = (a1_array[0] - a2_array[0]) * (b1_array[1] - a1_array[1]) + (a1_array[1] - a2_array[1]) * (a1_array[0] - b1_array[0])
  t2 = (a1_array[0] - a2_array[0]) * (b2_array[1] - a1_array[1]) + (a1_array[1] - a2_array[1]) * (a1_array[0] - b2_array[0])
  t3 = (b1_array[0] - b2_array[0]) * (a1_array[1] - b1_array[1]) + (b1_array[1] - b2_array[1]) * (b1_array[0] - a1_array[0])
  t4 = (b1_array[0] - b2_array[0]) * (a2_array[1] - b1_array[1]) + (b1_array[1] - b2_array[1]) * (b1_array[0] - a2_array[0])
  return t1 * t2 <= 0 and t3 * t4 <= 0

# 人差し指と中指が交差しているかどうか
def is_intersect_to_index_and_middle(hand_world_landmarks:landmark_pb2.LandmarkList):
  return int( __is_intersect(hand_world_landmarks.landmark[7], hand_world_landmarks.landmark[8], hand_world_landmarks.landmark[11], hand_world_landmarks.landmark[12]) or \
              __is_intersect(hand_world_landmarks.landmark[6], hand_world_landmarks.landmark[7], hand_world_landmarks.landmark[10], hand_world_landmarks.landmark[11])  )

# バウンティングボックスの左上，右下の座標を返す
def calc_bounding_rect(image, hand_landmarks:landmark_pb2.LandmarkList):
  height, width = image.shape[:2]

  landmark_array = np.empty((0, 2), int)

  for _, landmark in enumerate(hand_landmarks.landmark):
      landmark_x = min(int(landmark.x * width), width - 1)
      landmark_y = min(int(landmark.y * height), height - 1)

      landmark_point = [np.array((landmark_x, landmark_y))]

      landmark_array = np.append(landmark_array, landmark_point, axis=0)

  x, y, w, h = cv2.boundingRect(landmark_array)
  r = max(w,h)
  margin = 30
  return [x-margin, y-margin, x + w + margin, y + h + margin]

# バウンティングボックスの左上の座標を返す
def get_bounding_rect_top_left(image, hand_landmarks:landmark_pb2.LandmarkList):
  pos = calc_bounding_rect(image, hand_landmarks)
  return pos[:2]

def output_log(
  hand_landmarks:landmark_pb2.NormalizedLandmarkList, 
  hand_world_landmarks:landmark_pb2.LandmarkList, 
  handedness:classification_pb2.ClassificationList
):
  print(f'Hands[{id}]{{\n'
        f'Angles(Raw){calc_joint_angles(hand_world_landmarks)}\n'
        f'Direction: {calc_direction_angle(hand_landmarks)}\n'
        f'Hand_plam (1 is face): {is_plam_facing(hand_landmarks, handedness)}\n'
        f'Distance (Thumb & Index): {calc_distance_to_thumb_and_index_finger(hand_world_landmarks)}\n'
        f'Distance (Thumb & Middle): {calc_distance_to_thumb_and_middle_finger(hand_world_landmarks)}\n'
        f'Distance (Index & Middle): {calc_distance_to_index_and_middle_finger(hand_world_landmarks)}\n'
        f'Instersect: {is_intersect_to_index_and_middle(hand_world_landmarks)}\n'
        f'}}')

def to_csv(
  hand_landmarks:landmark_pb2.NormalizedLandmarkList, 
  hand_world_landmarks:landmark_pb2.LandmarkList, 
  handedness:classification_pb2.ClassificationList
):
  out_csv = ''
  for angle in calc_joint_angles(hand_world_landmarks):
    out_csv += f'{str(angle)}, '
  out_csv += f'{str(calc_direction_angle(hand_world_landmarks))}, ' 
  out_csv += f'{str(is_plam_facing(hand_landmarks, handedness))}, '
  out_csv += f'{str(calc_distance_to_thumb_and_index_finger(hand_world_landmarks))}, '
  out_csv += f'{str(calc_distance_to_thumb_and_middle_finger(hand_world_landmarks))}, '
  out_csv += f'{str(calc_distance_to_index_and_middle_finger(hand_world_landmarks))}, '
  out_csv += f'{str(is_intersect_to_index_and_middle(hand_world_landmarks))}\n'
  return out_csv

def get_explanatory_variables(
  hand_landmarks:landmark_pb2.NormalizedLandmarkList, 
  hand_world_landmarks:landmark_pb2.LandmarkList, 
  handedness:classification_pb2.ClassificationList
):
  explanatory_variables = []
  explanatory_variables.extend(calc_joint_angles(hand_world_landmarks))
  explanatory_variables.append(calc_direction_angle(hand_world_landmarks))
  explanatory_variables.append(is_plam_facing(hand_landmarks, handedness))
  explanatory_variables.append(calc_distance_to_thumb_and_index_finger(hand_world_landmarks))
  explanatory_variables.append(calc_distance_to_thumb_and_middle_finger(hand_world_landmarks))
  explanatory_variables.append(calc_distance_to_index_and_middle_finger(hand_world_landmarks))
  explanatory_variables.append(is_intersect_to_index_and_middle(hand_world_landmarks))
  return explanatory_variables