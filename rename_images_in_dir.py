import sys
import os

def rename_images_in_directory(directory_path):
    """
    주어진 디렉토리 내의 이미지 파일들을 <디렉토리명>_<번호>.<확장자> 형태로 순차적으로 이름 변경합니다.
    번호는 001부터 시작하며, 이미지 파일 확장자는 원본 그대로 사용합니다.
    
    예: a.png -> <디렉토리명>_001.png
    """
    # 디렉토리 이름 추출 (경로의 마지막 폴더명)
    dir_name = os.path.basename(os.path.normpath(directory_path))
    
    # 이미지 파일 확장자 목록 (필요에 따라 확장 가능)
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
    
    # 디렉토리 내의 이미지 파일 목록 추출 (대소문자 무시)
    image_files = [f for f in os.listdir(directory_path) if f.lower().endswith(valid_extensions)]
    
    # 순서를 위해 파일명 정렬 (알파벳 순)
    image_files.sort()
    
    # 파일 개수에 따라 001, 002, ... 순서대로 번호 부여하며 이름 변경
    for idx, filename in enumerate(image_files, start=1):
        # 파일 확장자 분리
        name, ext = os.path.splitext(filename)
        new_filename = f"{dir_name}_{idx:03d}{ext}"
        old_file = os.path.join(directory_path, filename)
        new_file = os.path.join(directory_path, new_filename)
        os.rename(old_file, new_file)
        print(f"{filename} -> {new_filename}")

# 사용 예시
if __name__ == '__main__':
    directory = sys.argv[1]#input("이미지 파일들이 있는 디렉토리 경로를 입력하세요: ").strip()
    rename_images_in_directory(directory)

