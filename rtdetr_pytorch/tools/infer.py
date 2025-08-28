import torch
import torch.nn as nn 
import torchvision.transforms as T
from torch.cuda.amp import autocast
import numpy as np 
from PIL import Image, ImageDraw, ImageFont
import os 
import sys 
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import argparse
import src.misc.dist as dist 
from src.core import YAMLConfig 
from src.solver import TASKS
import numpy as np
import cv2

COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus",
    "truck", "traffic light", "fire hydrant", "stop sign",
    "parking meter"
]

CLASS_COLORS = {
    "person": (0, 255, 0),          # Green
    "bicycle": (255, 0, 0),         # Red
    "car": (0, 0, 255),             # Blue
    "motorcycle": (255, 165, 0),    # Orange
    "airplane": (128, 0, 128),      # Purple
    "bus": (255, 255, 0),           # Yellow
    "truck": (0, 255, 255),         # Cyan
    "traffic light": (255, 0, 255), # Magenta
    "fire hydrant": (0, 128, 128),  # Teal
    "stop sign": (128, 0, 0),       # Dark Red
    "parking meter": (128, 128, 0)  # Olive
}

def postprocess(labels, boxes, scores, iou_threshold=0.55):
    def calculate_iou(box1, box2):
        x1, y1, x2, y2 = box1
        x3, y3, x4, y4 = box2
        xi1 = max(x1, x3)
        yi1 = max(y1, y3)
        xi2 = min(x2, x4)
        yi2 = min(y2, y4)
        inter_width = max(0, xi2 - xi1)
        inter_height = max(0, yi2 - yi1)
        inter_area = inter_width * inter_height
        box1_area = (x2 - x1) * (y2 - y1)
        box2_area = (x4 - x3) * (y4 - y3)
        union_area = box1_area + box2_area - inter_area
        iou = inter_area / union_area if union_area != 0 else 0
        return iou
    merged_labels = []
    merged_boxes = []
    merged_scores = []
    used_indices = set()
    for i in range(len(boxes)):
        if i in used_indices:
            continue
        current_box = boxes[i]
        current_label = labels[i]
        current_score = scores[i]
        boxes_to_merge = [current_box]
        scores_to_merge = [current_score]
        used_indices.add(i)
        for j in range(i + 1, len(boxes)):
            if j in used_indices:
                continue
            if labels[j] != current_label:
                continue  
            other_box = boxes[j]
            iou = calculate_iou(current_box, other_box)
            if iou >= iou_threshold:
                boxes_to_merge.append(other_box.tolist())  
                scores_to_merge.append(scores[j])
                used_indices.add(j)
        xs = np.concatenate([[box[0], box[2]] for box in boxes_to_merge])
        ys = np.concatenate([[box[1], box[3]] for box in boxes_to_merge])
        merged_box = [np.min(xs), np.min(ys), np.max(xs), np.max(ys)]
        merged_score = max(scores_to_merge)
        merged_boxes.append(merged_box)
        merged_labels.append(current_label)
        merged_scores.append(merged_score)
    return [np.array(merged_labels)], [np.array(merged_boxes)], [np.array(merged_scores)]
def slice_image(image, slice_height, slice_width, overlap_ratio):
    img_width, img_height = image.size
    
    slices = []
    coordinates = []
    step_x = int(slice_width * (1 - overlap_ratio))
    step_y = int(slice_height * (1 - overlap_ratio))
    
    for y in range(0, img_height, step_y):
        for x in range(0, img_width, step_x):
            box = (x, y, min(x + slice_width, img_width), min(y + slice_height, img_height))
            slice_img = image.crop(box)
            slices.append(slice_img)
            coordinates.append((x, y))
    return slices, coordinates
def merge_predictions(predictions, slice_coordinates, orig_image_size, slice_width, slice_height, threshold=0.30):
    merged_labels = []
    merged_boxes = []
    merged_scores = []
    orig_height, orig_width = orig_image_size
    for i, (label, boxes, scores) in enumerate(predictions):
        x_shift, y_shift = slice_coordinates[i]
        scores = np.array(scores).reshape(-1)
        valid_indices = scores > threshold
        valid_labels = np.array(label).reshape(-1)[valid_indices]
        valid_boxes = np.array(boxes).reshape(-1, 4)[valid_indices]
        valid_scores = scores[valid_indices]
        for j, box in enumerate(valid_boxes):
            box[0] = np.clip(box[0] + x_shift, 0, orig_width)  
            box[1] = np.clip(box[1] + y_shift, 0, orig_height)
            box[2] = np.clip(box[2] + x_shift, 0, orig_width)  
            box[3] = np.clip(box[3] + y_shift, 0, orig_height) 
            valid_boxes[j] = box
        merged_labels.extend(valid_labels)
        merged_boxes.extend(valid_boxes)
        merged_scores.extend(valid_scores)
    return np.array(merged_labels), np.array(merged_boxes), np.array(merged_scores)

def draw(images, labels, boxes, scores, threshold=0.6, save_path=""):
    """Draw bounding boxes with high-contrast colors and adaptive text color."""

    for i, img in enumerate(images):
        draw = ImageDraw.Draw(img)
        scr = scores[i]
        valid_indices = scr > threshold
        lab, box, scrs = labels[i][valid_indices], boxes[i][valid_indices], scr[valid_indices]
        font = ImageFont.load_default()

        for j, b in enumerate(box):
            class_id = int(lab[j].item())
            class_name = COCO_CLASSES[class_id] if class_id < len(COCO_CLASSES) else f"id:{class_id}"
            color = CLASS_COLORS.get(class_name, (255, 255, 255))  # Default: white

            # Draw bounding box (thickness reduced to 2)
            draw.rectangle(list(b), outline=color, width=2)

            # Label text
            conf = round(scrs[j].item(), 2)
            label_text = f"{class_name} {conf}"

            # Measure text box
            text_bbox = draw.textbbox((b[0], b[1]), label_text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]

            # Compute brightness of box color
            r, g, b_col = color
            brightness = 0.299 * r + 0.587 * g + 0.114 * b_col
            text_color = (0, 0, 0) if brightness > 128 else (255, 255, 255)

            # Draw filled background for label
            draw.rectangle([b[0], b[1] - text_height - 4, b[0] + text_width + 4, b[1]], fill=color)

            # Draw text with adaptive color
            draw.text((b[0] + 2, b[1] - text_height - 2), label_text, fill=text_color, font=font)

        img.save(save_path if save_path else f"results_{i}.jpg")

            
def main(args, ):
    """main
    """
    cfg = YAMLConfig(args.config, resume=args.resume)
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu') 
        if 'ema' in checkpoint:
            state = checkpoint['ema']['module']
        else:
            state = checkpoint['model']
    else:
        raise AttributeError('Only support resume to load model.state_dict by now.')
    # NOTE load train mode state -> convert to deploy mode
    cfg.model.load_state_dict(state)
    class Model(nn.Module):
        def __init__(self, ) -> None:
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()
            
        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            outputs = self.postprocessor(outputs, orig_target_sizes)
            return outputs
    
    model = Model().to(args.device)
    model.eval()  
    transforms = T.Compose([T.Resize((640, 640)), T.ToTensor()])  

    # VIDEO MODE
    if args.video:  
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            print(" Error: Could not open video.")
            return

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        out = cv2.VideoWriter("output_video.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            im_pil = Image.fromarray(img_rgb)
            w, h = im_pil.size
            orig_size = torch.tensor([w, h])[None].to(args.device)
            im_data = transforms(im_pil)[None].to(args.device)

            with torch.no_grad():
                labels, boxes, scores = model(im_data, orig_size)
            labels, boxes, scores = labels.cpu(), boxes.cpu(), scores.cpu()

            draw([im_pil], [labels], [boxes], [scores], 0.6, save_path="temp.jpg")
            annotated = cv2.imread("temp.jpg")
            out.write(annotated)

        cap.release()
        out.release()
        print(" Video inference completed. Saved as: output_video.mp4")
        return

    # IMAGE MODE 
    if args.im_file:
        im_pil = Image.open(args.im_file).convert('RGB')
        w, h = im_pil.size
        orig_size = torch.tensor([w, h])[None].to(args.device)
        im_data = transforms(im_pil)[None].to(args.device)

        if args.sliced:
            num_boxes = args.numberofboxes
            aspect_ratio = w / h
            num_cols = int(np.sqrt(num_boxes * aspect_ratio)) 
            num_rows = int(num_boxes / num_cols)
            slice_height = h // num_rows
            slice_width = w // num_cols
            overlap_ratio = 0.2
            slices, coordinates = slice_image(im_pil, slice_height, slice_width, overlap_ratio)
            predictions = []
            for i, slice_img in enumerate(slices):
                slice_tensor = transforms(slice_img)[None].to(args.device)
                with autocast():
                    output = model(slice_tensor, torch.tensor([[slice_img.size[0], slice_img.size[1]]]).to(args.device))
                torch.cuda.empty_cache()
                labels, boxes, scores = output
                labels = labels.cpu().detach().numpy()
                boxes = boxes.cpu().detach().numpy()
                scores = scores.cpu().detach().numpy()
                predictions.append((labels, boxes, scores))

            merged_labels, merged_boxes, merged_scores = merge_predictions(predictions, coordinates, (h, w), slice_width, slice_height)
            labels, boxes, scores = postprocess(merged_labels, merged_boxes, merged_scores)
        else:
            output = model(im_data, orig_size)
            labels, boxes, scores = output

        draw([im_pil], labels, boxes, scores, 0.6)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str)
    parser.add_argument('-r', '--resume', type=str)
    parser.add_argument('-f', '--im-file', type=str, default=None)  # MODIFIED to allow None
    parser.add_argument('--video', type=str, default=None, help="Path to input video")  # NEW
    parser.add_argument('-s', '--sliced', type=bool, default=False)
    parser.add_argument('-d', '--device', type=str, default='cpu')
    parser.add_argument('-nc', '--numberofboxes', type=int, default=25)
    args = parser.parse_args()
    main(args)