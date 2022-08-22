# Fingerspelling Recognition using [MediaPipe](https://github.com/google/mediapipe)
[MediaPipe Hands](https://google.github.io/mediapipe/solutions/hands)を用いて手の姿勢推定を行い、
取得した手の形状パラメータを用いて機械学習で、かな指文字を分類するプログラムです。  

## Introduction
### 環境
- Ubuntu 20.04 LTS  
- Windows 10 Home  
- Mac OS (未検証)  

### 必要なパッケージをインストール
```bash
pip install mediapipe tensorflow sklearn opencv-contrib-python
```
※ `mediapipe`はバージョン0.8.9以降のものを使用すること

## Demo
### リアルタイム指文字認識  
```bash
python src/fingerspelling_recognition.py
```
<img src="resource/fingerspelling_recognition_demo1.gif" width=480><br>  
* 「ESC」キー押下でアプリケーション終了

### 指文字によるテキスト入力  
```bash
python src/text_input_with_fingerspelling_recognition.py
```
<img src="resource/fingerspelling_recognition_demo2.gif" width=480><br>  
* 「ESC」キー押下でアプリケーション終了

## Feature Value（特徴量）  
MediaPipeから取得した座標情報を、以下の情報に変換し指文字分類モデルに流し込んでいます。
| 特徴量ID | 内容                           |
| :------: | :----------------------------- |
|    0     | 親指のCM関節の角度             |
|    1     | 親指のMP関節の角度             |
|    2     | 親指のIP関節の角度             |
|    3     | 人差し指のMP関節の角度         |
|    4     | 人差し指のPIP関節の角度        |
|    5     | 人差し指のDIP関節の角度        |
|    6     | 中指のMP関節の角度             |
|    7     | 中指のPIP関節の角度            |
|    8     | 中指のDIP関節の角度            |
|    9     | 薬指のMP関節の角度             |
|    10    | 薬指のDIP関節の角度            |
|    11    | 小指のMP関節の角度             |
|    12    | 小指のPIP関節の角度            |
|    13    | 小指のDIP関節の角度            |
|    14    | 手のロール角度                 |
|    15    | 掌の向き                       |
|    16    | 人差し指と中指の指先同士の距離 |
|    17    | 人差し指と薬指の指先同士の距離 |
|    18    | 中指と薬指の指先同士の距離     |
|    19    | 人差し指と中指が交差しているか |

`hand_utils.get_explanatory_variables(hand_landmarks, hand_world_landmarks, handedness)`で特徴量リストを取得できます。

## Reference  
- [google/Mediapipe](https://github.com/google/Mediapipe)  
- [Kazuhito00/hand-gesture-recognition-using-mediapipe](https://github.com/Kazuhito00/hand-gesture-recognition-using-mediapipe)（分類モデルの構築に参考させて頂きました）  

## Author  
Redgum775

## License  
[Apache v2 license](LICENSE)
