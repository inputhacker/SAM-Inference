import cv2
import torch
import time
import numpy as np
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamPredictor

# setting 
#sam_checkpoint = "sam_vit_h_4b8939.pth"
#model_type = "vit_h"
sam_checkpoint = "sam_vit_l_0b3195.pth"
model_type = "vit_l"
#sam_checkpoint = "sam_vit_b_01ec64.pth"
#model_type = "vit_b"
device = "cuda:0"

# utils
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))

# Mouse callback
def draw_point_and_box(event, x, y, flags, param):
    global point_mode, drawing_box, ix, iy, input_points, input_labels, input_boxes, image_copy, drawing
    # Point mode
    # positive / negative points
    if point_mode:
        if event == cv2.EVENT_LBUTTONDOWN:  
            # Left Mouse Click -> green(positive) point
            cv2.circle(image, (x, y), 5, (0, 255, 0), -1)  
            
            # Save points and label 
            input_points.append([x, y])  
            input_labels.append(1)
            
        elif event == cv2.EVENT_RBUTTONDOWN:  
            # Right Mouse Click -> red(negative) point
            cv2.circle(image, (x, y), 5, (0, 0, 255), -1) 
            
            # Save points and label 
            input_points.append([x, y])  
            input_labels.append(0)
            
        cv2.imshow('Image', image)
    # Bbox mode   
    elif drawing_box:
        if event == cv2.EVENT_LBUTTONDOWN:  
            ix, iy = x, y
            drawing = True
            
        elif event == cv2.EVENT_MOUSEMOVE:  
            if drawing:
                temp_image = image_copy.copy()
                cv2.rectangle(temp_image, (ix, iy), (x, y), (0, 255, 0), 1)  
                cv2.imshow('Image', temp_image)
                
        elif event == cv2.EVENT_LBUTTONUP:  
            drawing = False
            cv2.rectangle(image, (ix, iy), (x, y), (0, 255, 0), 1)  
            
            # save points (x, y, x y)
            input_boxes.append([ix, iy, x, y])  
            cv2.imshow('Image', image)

def find_largest_and_best_boundingbox(mask):
    # 채널 축 제거 (1, H, W) → (H, W)
    mask = mask.squeeze(0).astype(np.uint8) * 255  # 0 또는 255로 변환

    # 컨투어 탐색 (cv2.RETR_EXTERNAL: 외곽 컨투어만)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:  # 컨투어가 없을 경우
        return None, None

    # (1) 가장 큰 개별 contour 찾기
    largest_contour = max(contours, key=cv2.contourArea)  # 면적 기준 최대 contour
    largest_box = cv2.boundingRect(largest_contour)  # (x, y, width, height)

    # (2) 모든 contour를 포함하는 box_best 찾기
    all_points = np.concatenate(contours)  # 모든 컨투어 점들을 하나의 배열로 합침
    best_box = cv2.boundingRect(all_points)  # 전체 컨투어 포함하는 박스

    # Largest box 표시
    cv2.rectangle(bb_copy, (largest_box[0], largest_box[1]),
                            (largest_box[0]+largest_box[2], largest_box[1]+largest_box[3]), (0,0,255), 3)
    # Best box 표시
    cv2.rectangle(bb_copy, (best_box[0], best_box[1]),
                            (best_box[0]+best_box[2], best_box[1]+best_box[3]), (255,0,0), 2)

    # Convex hull 표시
    all_hull_points = []
    for contour in contours:
        hull = cv2.convexHull(contour)
        all_hull_points.extend(hull)  # 모든 Hull의 점들을 저장

    # 모든 Convex Hull을 포함하는 최적의 바운딩 박스
    all_hull_points = np.array(all_hull_points)
    hull_bb = cv2.boundingRect(all_hull_points)
    cv2.rectangle(bb_copy, (hull_bb[0], hull_bb[1]),
                            (hull_bb[0]+hull_bb[2], hull_bb[1]+hull_bb[3]), (0,255,0), 1)

    print(f"largest box[] = {largest_box} <- Red")
    print(f"best box[] = {best_box} <- Blue")
    print(f"hull_bb[] = {hull_bb} <- Green")
    cv2.imshow('Largest box, best box and hull box', bb_copy)

    return largest_box, best_box

def main():
    # SAM 
    print('Loading Segment Anything Model...')
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)
   
    while True:
        # input image path
        img_path = input("Image path: ")
        print()

        # input image & callbacks
        global image, point_mode, drawing_box, ix, iy, input_points, input_labels, input_boxes, image_copy, bb_copy, drawing
        
        image = cv2.imread(img_path)
        if image is None:
            print('Error: No such file or directory.\n')
            continue             
        
        # global
        point_mode = False
        drawing_box = False
        drawing = False
        ix, iy = -1, -1
        input_points = []
        input_labels = []
        input_boxes = []
        image_copy = image.copy()
        bb_copy = image_copy.copy()
        
        predictor.set_image(image) 
        
        print('------')
        print('Segmentation Mode List')
        print('s: Single Object Segmentation / m: Multiple Object Segmentation / q: Quit')
        print('------\n')
        
        cv2.imshow('Image', image)
        cv2.setMouseCallback('Image', draw_point_and_box)  
        
        while True:
            key = cv2.waitKey(1)
            
            if key & 0xFF == ord('q'):
                break
            
            elif key & 0xFF == ord('s'):
                print('---Sinlge Object Segmentation Mode---')
                print('Support \n 1. Multiple points \n 2. Single bbox\n')
                print('press the keyboard')
                print('p: Point Mode / b: Bbox Mode / r: Reset / i: Model Inference / q: Quit\n')
                
                while True: 
                    key2 = cv2.waitKey(1)
                
                    if key2 & 0xFF == ord('q'):
                        print('---Segmentation Mode Change---')
                        print('s: Single Object Segmentation / m: Multiple Object Segmentation / q: Quit\n')
                        input_points = []  
                        input_labels = []
                        input_boxes = [] 
                        drawing_box = False
                        point_mode = False
                        image = image_copy.copy()
                        cv2.imshow('Image', image)
                        break                
                    
                    elif key2 & 0xFF == ord('p'):
                        point_mode = True
                        drawing_box = False
                        print('Now: Point Mode')
                        print('p: Point Mode / b: Bbox Mode / r: Reset / i: Model Inference / q: Quit\n')
                        
                    elif key2 & 0xFF == ord('b'):
                        drawing_box = True
                        point_mode = False
                        print('Now: Bbox Mode')
                        print('p: Point Mode / b: Bbox Mode / r: Reset / i: Model Inference / q: Quit\n')
                        
                    elif key2 & 0xFF == ord('r'):
                        # reset 
                        input_points = []  
                        input_labels = []
                        input_boxes = [] 
                        drawing_box = False
                        point_mode = False
                        image = image_copy.copy()
                        cv2.imshow('Image', image)
                        print('Now: Reset all points and boxes')
                        print('p: Point Mode / b: Bbox Mode / r: Reset / i: Model Inference / q: Quit\n')
                        
                    elif key2 & 0xFF == ord('i'):
                        print('Now: Model Inference')
                        print('p: Point Mode / b: Bbox Mode / r: Reset / i: Model Inference / q: Quit\n')
                        input_points_sam = np.array(input_points)
                        input_labels_sam = np.array(input_labels)
                        input_boxes_sam = np.array(input_boxes)
                        
                        if input_boxes_sam.size > 5:
                            print('Error: Only one bbox is allowed')
                            print('p: Point Mode / b: Bbox Mode / r: Reset / i: Model Inference / q: Quit\n')
                            continue
                        
                        is_empty_points = input_labels_sam.size == 0
                        is_empty_boxes = input_boxes_sam.size == 0
                        
                        if is_empty_points:
                            input_points_sam = None
                            input_labels_sam = None
                        
                        if is_empty_boxes:
                            input_boxes_sam = None
                            
                        if is_empty_boxes and is_empty_points:
                            print('Error: No Input Prompt')
                            print('p: Point Mode / b: Bbox Mode / r: Reset / i: Model Inference / q: Quit\n')
                            continue
                                
                        masks, _, _ = predictor.predict(
                            point_coords=input_points_sam,
                            point_labels=input_labels_sam,
                            box=input_boxes_sam,
                            multimask_output=False,)

                        bb_copy = image_copy.copy()
                        largest_bb, best_bb = find_largest_and_best_boundingbox(masks)

                        if largest_bb == None or best_bb == None:
                            print(f"largest_bb or best_bb is None")
                            continue

                        plt.figure(num='Segmentation Result', figsize=(10, 10))
                        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                        show_mask(masks[0], plt.gca())
                        
                        if not is_empty_boxes:
                            show_box(input_boxes_sam[0], plt.gca())
                            
                        if not is_empty_points:
                            show_points(input_points_sam, input_labels_sam, plt.gca())

                        plt.axis('off')
                        # save image 
                        current_time = time.strftime("%y%m%d_%H%M%S")
                        image_name = f"results/Result_{current_time}.png"
                        plt.savefig(image_name)                        
                        plt.show()                    
                
                
            elif key & 0xFF == ord('m'):
                print('---Multiple Object Segmentation---')
                print('Support \n 1. Multiple bboxes \n')
                print('press the keyboard')
                print('b: Bbox Mode / r: Reset / i: Model Inference / q: Quit\n')

                while True:
                    key2 = cv2.waitKey(1)
                    
                    if key2 & 0xFF == ord('q'):
                        print('---Segmentation Mode Change---')
                        print('s: Single Object Segmentation / m: Multiple Object Segmentation / q: Quit\n')
                        input_points = []  
                        input_labels = []
                        input_boxes = [] 
                        drawing_box = False
                        point_mode = False
                        image = image_copy.copy()
                        cv2.imshow('Image', image)                        
                        break                    

                    elif key2 & 0xFF == ord('b'):
                        drawing_box = True
                        point_mode = False
                        print('Now: Bbox Mode')
                        print('b: Bbox Mode / r: Reset / i: Model Inference / q: Quit\n')
                        
                    elif key2 & 0xFF == ord('r'):
                        # reset 
                        input_points = []  
                        input_labels = []
                        input_boxes = [] 
                        drawing_box = False
                        point_mode = False
                        image = image_copy.copy()
                        cv2.imshow('Image', image)
                        print('Now: Reset all points and boxes')
                        print('b: Bbox Mode / r: Reset / i: Model Inference / q: Quit\n')
                        
                    elif key2 & 0xFF == ord('i'):
                        print('Now: Model Inference')
                        print('b: Bbox Mode / r: Reset / i: Model Inference / q: Quit\n')
                        
                        input_boxes_sam = torch.tensor(input_boxes, device=predictor.device)
                        
                        if input_boxes_sam.shape[0] == 0:
                            print('Error: No Input Prompt')
                            print('b: Bbox Mode / r: Reset / i: Model Inference / q: Quit\n')
                            continue
                        
                        transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes_sam, image.shape[:2])
                        
                        masks, _, _ = predictor.predict_torch(
                            point_coords=None,
                            point_labels=None,
                            boxes=transformed_boxes,
                            multimask_output=False,)
                        
                        plt.figure(num='Segmentation Result', figsize=(10, 10))
                        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                        
                        for mask in masks:
                            show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
                        for box in input_boxes_sam:
                            show_box(box.cpu().numpy(), plt.gca())

                        plt.axis('off')
                        # save image 
                        current_time = time.strftime("%y%m%d_%H%M%S")
                        image_name = f"results/Result_{current_time}.png"
                        plt.savefig(image_name)
                        plt.show()

        cv2.destroyAllWindows()

        is_continue = input("Continue (y/n): ")
        if is_continue == 'y':
            print()
            continue
        else:
            print('---Exit---')
            break
        
if __name__ == '__main__':
    main()
