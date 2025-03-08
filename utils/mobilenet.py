import sys
import cv2
import numpy as np
import tensorflow as tf
import time
import matplotlib.pyplot as plt

# TFLite 모델 로드 및 인터프리터 초기화
interpreter = tf.lite.Interpreter(model_path="ssd_mobilenet_v2.tflite")
interpreter.allocate_tensors()

# 입력/출력 텐서 정보 얻기
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 입력 이미지 불러오기 및 전처리
img = cv2.imread(sys.argv[1])#"image.jpg")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
input_shape = input_details[0]['shape'][1:3]
img_resized = cv2.resize(img_rgb, (input_shape[1], input_shape[0]))
input_data = np.expand_dims(img_resized, axis=0)
# 모델에 따라 정규화/데이터타입 변경 (여기서는 uint8로 가정)
input_data = input_data.astype(np.uint8)

# 추론 수행 및 시간 측정
start_time = time.time()
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()
inference_time = (time.time() - start_time) * 1000  # ms 단위
print(f"Inference time: {inference_time:.2f} ms")

# 출력 텐서에서 결과 추출 (모델에 따라 인덱스는 달라질 수 있음)
boxes = interpreter.get_tensor(output_details[0]['index'])[0]      # [N, 4]
classes = interpreter.get_tensor(output_details[1]['index'])[0]    # [N]
scores = interpreter.get_tensor(output_details[2]['index'])[0]     # [N]
num_detections = int(interpreter.get_tensor(output_details[3]['index'])[0])

# 결과 시각화 (score threshold 설정)
score_threshold = 0.5
h, w, _ = img_rgb.shape
for i in range(num_detections):
    if scores[i] < score_threshold:
        continue
    # 모델 출력은 보통 [ymin, xmin, ymax, xmax] 형태이므로 좌표 변환
    ymin, xmin, ymax, xmax = boxes[i]
    x1, y1 = int(xmin * w), int(ymin * h)
    x2, y2 = int(xmax * w), int(ymax * h)
    cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)

plt.figure(figsize=(10, 10))
plt.imshow(img_rgb)
plt.title("TFLite SSD MobileNet V2 Detection")
plt.axis("off")
plt.show()

