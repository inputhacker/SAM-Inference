import sys
import cv2
import matplotlib.pyplot as plt

# 이미지 로드 (이미지 경로를 적절히 수정)
img = cv2.imread(sys.argv[1])#"image.jpg")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Selective Search 초기화
ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
ss.setBaseImage(img)
# 속도와 정확도의 trade-off: 빠른 방식(fast) vs. 더 정밀한 방식(quality)
#ss.switchToSelectiveSearchFast()  # 보다 빠른 처리. 더 높은 품질이 필요하면 switchToSelectiveSearchQuality() 사용
ss.switchToSelectiveSearchQuality()

# 후보 영역 추출
rects = ss.process()
print("전체 후보 영역 수:", len(rects))

# 상위 N개의 후보 영역 선택 (예: N=100)
N = 5
boxes = rects[:N]

# 후보 영역 시각화
img_out = img_rgb.copy()
for (x, y, w, h) in boxes:
    cv2.rectangle(img_out, (x, y), (x + w, y + h), (0, 255, 0), 2)

plt.figure(figsize=(10, 10))
plt.imshow(img_out)
plt.title("Selective Search 후보 영역")
plt.axis("off")
plt.show()

