import cv2
import os
import numpy as np

def process_image(image_path):
    """
    주어진 이미지 경로에 대해 이미지를 불러오고,
    가장 작은 contour와 가장 큰 contour의 bounding box를 각각 붉은색과 분홍색으로 표시한 결과 이미지를 반환합니다.
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"이미지를 불러올 수 없습니다: {image_path}")
        return None

    # 그레이스케일 변환 및 이진화 (임계값은 필요에 따라 조정)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # 외곽 contour 검출
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print(f"컨투어를 찾을 수 없습니다: {image_path}")
        return image

    # 각 contour의 면적을 기준으로 최소/최대 contour 선택
    min_contour = min(contours, key=cv2.contourArea)
    max_contour = max(contours, key=cv2.contourArea)

    # bounding box 계산
    x_min, y_min, w_min, h_min = cv2.boundingRect(min_contour)
    x_max, y_max, w_max, h_max = cv2.boundingRect(max_contour)

    # bounding box 그리기
    # 붉은색: BGR (0, 0, 255)
    cv2.rectangle(image, (x_min, y_min), (x_min + w_min, y_min + h_min), (0, 0, 255), 2)
    # 분홍색: RGB (255,192,203) → BGR (203,192,255)
    cv2.rectangle(image, (x_max, y_max), (x_max + w_max, y_max + h_max), (203, 192, 255), 2)

    return image

def get_image_files(directory):
    """
    주어진 디렉토리 내의 이미지 파일(확장자 기준)을 찾아 정렬된 리스트로 반환합니다.
    """
    valid_ext = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
    files = [os.path.join(directory, f) for f in os.listdir(directory)
             if os.path.splitext(f)[1].lower() in valid_ext]
    files.sort()
    return files

def main(directory):
    image_files = get_image_files(directory)
    if not image_files:
        print("디렉토리에 이미지 파일이 없습니다.")
        return

    idx = 0
    while True:
        image_path = image_files[idx]
        processed = process_image(image_path)
        if processed is None:
            # 이미지 로드에 실패한 경우 다음 이미지로 넘어감
            idx = (idx + 1) % len(image_files)
            continue

        # 현재 이미지 정보(파일명 및 순번)를 윈도우 타이틀에 표시
        window_title = f"{os.path.basename(image_path)} ({idx+1}/{len(image_files)})"
        cv2.imshow(window_title, processed)

        # 키 입력 대기: 좌측, 우측 화살표키 및 ESC(종료)
        key = cv2.waitKeyEx(0)

        cv2.destroyWindow(window_title)  # 다음 이미지를 위해 현재 윈도우 닫기

        # ESC키(27)가 눌리면 종료
        if key == 27:
            break

        # 운영체제에 따라 좌측, 우측 화살표키의 키코드가 다를 수 있음
        # 일반적으로 Windows: 좌측 2424832, 우측 2555904
        # Linux: 좌측 65361, 우측 65363 또는 81, 83
        if key in [81, 2424832, 65361]:       # 왼쪽 화살표키: 이전 이미지
            idx = (idx - 1) % len(image_files)
        elif key in [83, 2555904, 65363]:     # 오른쪽 화살표키: 다음 이미지
            idx = (idx + 1) % len(image_files)
        # 다른 키 입력 시 현재 이미지 유지

    cv2.destroyAllWindows()

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("사용법: python script.py 이미지_디렉토리_경로")
    else:
        main(sys.argv[1])

