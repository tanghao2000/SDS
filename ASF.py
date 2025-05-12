import os
from PIL import Image
import numpy as np
def extract_class_names_and_iou(base_mask_file, comp_mask_file):
    # for voc
    class_mapping = {
        0: 'background', 1: 'aeroplane', 2: 'bicycle', 3: 'bird', 4: 'boat', 5: 'bottle', 6: 'bus',
        7: 'car', 8: 'cat', 9: 'chair', 10: 'cow', 11: 'diningtable', 12: 'dog',
        13: 'horse', 14: 'motorbike', 15: 'person', 16: 'pottedplant', 17: 'sheep',
        18: 'sofa', 19: 'train', 20: 'tvmonitor', 255: 'border'
    }
    # for coco
    # class_mapping = {
    #     0: 'background', 1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane', 6: 'bus', 7: 'train', 8: 'truck',
    #     9: 'boat', 10: 'traffic light', 11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 15: 'bench',
    #     16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow', 22: 'elephant', 23: 'bear', 24: 'zebra',
    #     25: 'giraffe', 27: 'backpack', 28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase', 34: 'frisbee',
    #     35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite', 39: 'baseball bat', 40: 'baseball glove',
    #     41: 'skateboard', 42: 'surfboard', 43: 'tennis racket', 44: 'bottle', 46: 'wine glass', 47: 'cup',
    #     48: 'fork', 49: 'knife', 50: 'spoon', 51: 'bowl', 52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange',
    #     56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut', 61: 'cake', 62: 'chair', 63: 'couch',
    #     64: 'potted plant', 65: 'bed', 67: 'dining table', 70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse',
    #     75: 'remote', 76: 'keyboard', 77: 'cell phone', 78: 'microwave', 79: 'oven', 80: 'toaster', 81: 'sink',
    #     82: 'refrigerator', 84: 'book', 85: 'clock', 86: 'vase', 87: 'scissors', 88: 'teddy bear', 89: 'hair drier',
    #     90: 'toothbrush', 255: 'border'
    # }



    base_mask = Image.open(base_mask_file)
    comp_mask = Image.open(comp_mask_file)

    base_mask_np = np.array(base_mask)
    comp_mask_np = np.array(comp_mask)

    unique_classes = np.unique(base_mask_np)

    total_intersection = 0
    total_union = 0

    iou_values = {}
    for class_val in unique_classes:
        if class_val in [0, 255]:  # Skip 'background' and 'border' classes
            continue

        base_class_pixels = (base_mask_np == class_val)
        comp_class_pixels = (comp_mask_np == class_val)

        intersection = np.logical_and(base_class_pixels, comp_class_pixels)
        union = np.logical_or(base_class_pixels, comp_class_pixels)

        iou = np.sum(intersection) / np.sum(union)
        iou_values[class_mapping[class_val]] = iou

        total_intersection += np.sum(intersection)
        total_union += np.sum(union)

    mIoU = total_intersection / total_union
    return iou_values, mIoU

real_mask_path = "E:\\voc_gen\\mask"
syn_mask_path = "E:\\voc_gen\\mask_CLIP"
output_txt_path = "E:\\voc_gen\\train_PCS.txt"

# Read the file names from output.txt
with open(output_txt_path, 'r') as file:
    mask_filenames = [line.strip() for line in file]

# Create or open the file to store results
result_file_path = "E:\\voc_gen\\train_ASF.txt"
result_file = open(result_file_path, "a")  # 'a' means append mode, which allows adding results to an existing file if it exists

class_mapping = {
    0: 'background', 1: 'aeroplane', 2: 'bicycle', 3: 'bird', 4: 'boat', 5: 'bottle', 6: 'bus',
    7: 'car', 8: 'cat', 9: 'chair', 10: 'cow', 11: 'diningtable', 12: 'dog',
    13: 'horse', 14: 'motorbike', 15: 'person', 16: 'pottedplant', 17: 'sheep',
    18: 'sofa', 19: 'train', 20: 'tvmonitor', 255: 'border'
}

for num_classes in range(1, len(class_mapping) - 1):  # 从1个类别到倒数第二个类别
    result_file_path = f"E:\\voc_gen\\iou_results\\iou_results_{num_classes}.txt"
    result_file = open(result_file_path, "w")

    results_list = []

    for filename in mask_filenames:
        real_mask_file = os.path.join(real_mask_path, f"{filename}.png")
        syn_mask_file = os.path.join(syn_mask_path, f"{filename}.png")

        if os.path.exists(syn_mask_file):
            iou_values, mIoU = extract_class_names_and_iou(real_mask_file, syn_mask_file)

            if len(iou_values) == num_classes:
                results_list.append({'filename': filename, 'mIoU': mIoU, 'iou_values': iou_values})

    sorted_results = sorted(results_list, key=lambda x: x['mIoU'], reverse=True)


    for result in sorted_results:
        result_file.write(f"Mask: {result['filename']}, mIoU: {result['mIoU']}, IoU: {result['iou_values']}\n")
        # result_file.write(f"{result['filename']}\n")

    result_file.close()