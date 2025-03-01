import sys
import os
import cv2
import json
import shutil
import xml.etree.ElementTree as ET

def parse_bboxes(bb_file_path):
    """
    주어진 bounding box 텍스트 파일을 파싱합니다.
    각 줄은 아래 두 형식 중 하나라고 가정합니다.
      - "xmin ymin xmax ymax" → 기본 클래스 "object" 할당
      - "class_name xmin ymin xmax ymax"
    반환: 각 박스를 dict (키: 'class', 'bbox' - [xmin, ymin, xmax, ymax]) 형태로 저장한 리스트.
    """
    boxes = []
    with open(bb_file_path, 'r') as f:
        for line in f:
            tokens = line.strip().split()
            if not tokens:
                continue
            # 만약 첫 토큰이 숫자일 경우(박스 개수 등) 건너뛰기 가능
            if len(tokens) == 1 and tokens[0].isdigit():
                continue

            if len(tokens) == 4:
                try:
                    xmin, ymin, xmax, ymax = map(int, tokens)
                    boxes.append({'class': 'object', 'bbox': [xmin, ymin, xmax, ymax]})
                except ValueError:
                    continue
            elif len(tokens) == 5:
                class_name = tokens[0]
                try:
                    xmin, ymin, xmax, ymax = map(int, tokens[1:])
                    boxes.append({'class': class_name, 'bbox': [xmin, ymin, xmax, ymax]})
                except ValueError:
                    continue
            else:
                # 예상치 못한 포맷은 무시
                continue
    return boxes

def draw_bboxes_on_image(image_path, boxes, output_path):
    """
    이미지 파일을 읽어 각 bounding box를 서로 다른 색상의 사각형(두께 2)으로 그림.
    그리고 output_path에 저장.
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"이미지를 읽을 수 없습니다: {image_path}")
        return
    # 미리 정의한 색상 리스트 (필요 시 색상 수를 늘리거나 랜덤하게 생성 가능)
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
    for idx, box in enumerate(boxes):
        color = colors[idx % len(colors)]
        xmin, ymin, xmax, ymax = box['bbox']
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)
    cv2.imwrite(output_path, image)

def create_pascal_voc_annotation(image_file, boxes, output_xml_path):
    """
    주어진 이미지 파일과 bounding box 정보를 바탕으로 Pascal VOC 형식의 XML annotation 파일을 생성합니다.
    생성된 XML 파일은 output_xml_path에 저장됩니다.
    """
    annotation = ET.Element('annotation')
    
    filename_elem = ET.SubElement(annotation, 'filename')
    filename_elem.text = os.path.basename(image_file)
    
    # 각 bounding box에 대해 object 태그 생성
    for box in boxes:
        obj = ET.SubElement(annotation, 'object')
        name_elem = ET.SubElement(obj, 'name')
        name_elem.text = box['class']
        
        bndbox = ET.SubElement(obj, 'bndbox')
        xmin_elem = ET.SubElement(bndbox, 'xmin')
        xmin_elem.text = str(box['bbox'][0])
        ymin_elem = ET.SubElement(bndbox, 'ymin')
        ymin_elem.text = str(box['bbox'][1])
        xmax_elem = ET.SubElement(bndbox, 'xmax')
        xmax_elem.text = str(box['bbox'][2])
        ymax_elem = ET.SubElement(bndbox, 'ymax')
        ymax_elem.text = str(box['bbox'][3])
    
    tree = ET.ElementTree(annotation)
    tree.write(output_xml_path)

def process_dataset(input_dir, output_dir):
    """
    전체 데이터셋을 처리하는 함수입니다.
    
    동작:
      1. input_dir 내의 이미지 파일(확장자 .jpg, .jpeg, .png)과 대응되는 _bb.txt 파일을 읽음.
      2. 각 이미지에 대해:
         - _bb.txt 파일을 파싱하여 bounding box 정보를 획득.
         - 원본 이미지에 각 박스를 서로 다른 색상의 두께 2의 사각형으로 그리고, 이미지명_bboxes 파일로 저장.
         - 원본 이미지를 output_dir/data 폴더로 복사.
         - Pascal VOC 형식의 XML annotation 파일을 output_dir/Annotations 폴더에 생성.
         - COCO 형식의 annotation 정보를 수집.
      3. 모든 이미지 처리 후, output_dir/labels.json 파일에 COCO annotation 정보를 저장.
    
    최종 디렉토리 구조 예시:
      <output_dir>/
        data/          → 원본 이미지들이 위치
        Annotations/   → 각 이미지의 XML annotation 파일들
        labels.json    → COCO annotation JSON 파일
    """
    # 출력 디렉토리 내에 필요한 폴더 생성
    data_dir = os.path.join(output_dir, "data")
    data_bbox_dir = os.path.join(output_dir, "data_bbox")
    annotations_dir = os.path.join(output_dir, "Annotations")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(data_bbox_dir, exist_ok=True)
    os.makedirs(annotations_dir, exist_ok=True)
    
    # COCO annotation 정보를 위한 구조 준비
    coco_images = []
    coco_annotations = []
    categories = {}          # 클래스명 -> id 매핑
    coco_categories = []     # COCO 카테고리 리스트
    
    next_image_id = 0
    next_annotation_id = 0
    
    # input_dir 내의 이미지 파일 처리 (확장자: .jpg, .jpeg, .png)
    valid_extensions = ('.jpg', '.jpeg', '.png')
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(valid_extensions):
            image_path = os.path.join(input_dir, filename)
            base_name, ext = os.path.splitext(filename)
            bb_file = os.path.join(input_dir, base_name + "_bb.txt")
            if not os.path.exists(bb_file):
                print(f"Bounding box 파일이 존재하지 않습니다: {filename}")
                continue
            
            # _bb.txt 파일 파싱
            boxes = parse_bboxes(bb_file)
 
            # 원본 이미지에 bounding box(es)를 표시한 이미지를 output_dir/data_bbox로 복사
            dst_bbox_image_path = data_dir + "_bbox"
            #shutil.copy(image_path, dst_bbox_image_path)
            
            # 원본 이미지에 박스 그린 이미지 저장 (dst_bbox_image_path에 imageName_bboxes.<ext>로 저장)
            drawn_image_name = base_name + "_bboxes" + ext
            drawn_image_path = os.path.join(dst_bbox_image_path, drawn_image_name)
            print(f"dst_bbox_image_path={dst_bbox_image_path}")
            print(f"drawn_image_path={drawn_image_path}")
            draw_bboxes_on_image(image_path, boxes, drawn_image_path)
            
            # 원본 이미지를 output_dir/data로 복사 (COCO 및 Pascal VOC에서 사용)
            dst_image_path = os.path.join(data_dir, filename)
            shutil.copy(image_path, dst_image_path)
           
            # Pascal VOC XML annotation 파일 생성 (output_dir/Annotations 내에 저장)
            xml_output_path = os.path.join(annotations_dir, base_name + ".xml")
            create_pascal_voc_annotation(filename, boxes, xml_output_path)
            
            # COCO 이미지 정보 등록
            coco_image_info = {
                "id": next_image_id,
                "file_name": filename
            }
            coco_images.append(coco_image_info)
            
            # 각 bounding box에 대해 COCO annotation 정보 등록
            for box in boxes:
                class_name = box['class']
                # 클래스가 아직 등록되지 않았다면 새 id 할당
                if class_name not in categories:
                    new_id = len(categories) + 1
                    categories[class_name] = new_id
                    coco_categories.append({"id": new_id, "name": class_name})
                xmin, ymin, xmax, ymax = box['bbox']
                width = xmax - xmin
                height = ymax - ymin
                annotation_info = {
                    "id": next_annotation_id,
                    "image_id": next_image_id,
                    "category_id": categories[class_name],
                    "bbox": [xmin, ymin, width, height]
                }
                coco_annotations.append(annotation_info)
                next_annotation_id += 1
            
            next_image_id += 1
    
    # COCO annotation 최종 JSON 구조 생성
    coco_dict = {
        "categories": coco_categories,
        "images": coco_images,
        "annotations": coco_annotations
    }
    # output_dir에 labels.json으로 저장
    coco_output_path = os.path.join(output_dir, "labels.json")
    with open(coco_output_path, 'w') as f:
        json.dump(coco_dict, f, indent=4)
    
    print("데이터셋 생성이 완료되었습니다.")

if __name__ == '__main__':
    # 사용 예시
    input_directory = sys.argv[1]   # 원본 이미지 및 *_bb.txt 파일이 위치한 디렉토리
    output_directory = sys.argv[2] # Pascal VOC (Annotations, data) 및 COCO (labels.json) 형식으로 저장될 디렉토리
    process_dataset(input_directory, output_directory)

