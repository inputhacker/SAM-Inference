import sys
import torch
import torchvision
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 사전 학습된 Faster R-CNN 모델 로드
#model = torch.hub.load('pytorch/vision', 'fasterrcnn_resnet50_fpn', pretrained=True)

#model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
#model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=True)
model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=True)
model.eval()

# 이미지 로드 (이미지 경로를 적절히 수정)
img_pil = Image.open(sys.argv[1]).convert("RGB")
#img_pil = Image.open("image.jpg").convert("RGB")
transform = transforms.Compose([transforms.ToTensor()])
img_tensor = transform(img_pil)

# 추론 수행
with torch.no_grad():
    predictions = model([img_tensor])

# 일정 score 임계값 이상의 박스만 사용 (예: threshold=0.5)
threshold = 0.5
boxes = predictions[0]['boxes'][predictions[0]['scores'] > threshold].numpy()

# 결과 시각화
fig, ax = plt.subplots(1, figsize=(10, 10))
ax.imshow(img_pil)
for box in boxes:
    x1, y1, x2, y2 = box
    rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
ax.set_title("Faster R-CNN 후보 영역")
plt.axis("off")
plt.show()

