import os
import cv2
import json
import subprocess
import numpy as np
from ultralytics import YOLO
from collections import deque
from tqdm import tqdm

# Colors for visualization
COLORS = [(56, 56, 255), (151, 157, 255), (31, 112, 255), (29, 178, 255), (49, 210, 207), (10, 249, 72), (23, 204, 146),
          (134, 219, 61), (52, 147, 26), (187, 212, 0), (168, 153, 44), (255, 194, 0), (147, 69, 52), (255, 115, 100),
          (236, 24, 0), (255, 56, 132), (133, 0, 82), (255, 56, 203), (200, 149, 255), (199, 55, 255)]

# Initialize YOLO model
def load_model(model_path):
    return YOLO(model_path)

# Function to convert results to JSON
def result_to_json(result, tracker=None):
    len_results = len(result.boxes)
    result_list_json = [
        {
            'class_id': int(result.boxes.cls[idx]),
            'class': result.names[int(result.boxes.cls[idx])],
            'confidence': float(result.boxes.conf[idx]),
            'bbox': {
                'x_min': int(result.boxes.data[idx][0]),
                'y_min': int(result.boxes.data[idx][1]),
                'x_max': int(result.boxes.data[idx][2]),
                'y_max': int(result.boxes.data[idx][3]),
            },
        } for idx in range(len_results)
    ]
    if result.masks is not None:
        for idx in range(len_results):
            result_list_json[idx]['mask'] = cv2.resize(result.masks.data[idx].cpu().numpy(), (result.orig_shape[1], result.orig_shape[0])).tolist()
            result_list_json[idx]['segments'] = result.masks.xyn[idx].tolist()
    if result.keypoints is not None:
        for idx in range(len_results):
            result_list_json[idx]['keypoints'] = result.keypoints.xyn[idx].tolist()
    if tracker is not None:
        bbs = [
            (
                [
                    result_list_json[idx]['bbox']['x_min'],
                    result_list_json[idx]['bbox']['y_min'],
                    result_list_json[idx]['bbox']['x_max'] - result_list_json[idx]['bbox']['x_min'],
                    result_list_json[idx]['bbox']['y_max'] - result_list_json[idx]['bbox']['y_min']
                ],
                result_list_json[idx]['confidence'],
                result_list_json[idx]['class'],
            ) for idx in range(len_results)
        ]
        tracks = tracker.update_tracks(bbs, frame=result.orig_img)
        for idx in range(len(result_list_json)):
            track_idx = next((i for i, track in enumerate(tracks) if track.det_conf is not None and np.isclose(track.det_conf, result_list_json[idx]['confidence'])), -1)
            if track_idx != -1:
                result_list_json[idx]['object_id'] = int(tracks[track_idx].track_id)
    return result_list_json

# Function to visualize results
def view_result(result, result_list_json, centers=None):
    image = result.plot(labels=False, line_width=2)
    for res in result_list_json:
        class_color = COLORS[res['class_id'] % len(COLORS)]
        text = f"{res['class']} {res['object_id']}: {res['confidence']:.2f}" if 'object_id' in res else f"{res['class']}: {res['confidence']:.2f}"
        (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
        cv2.rectangle(image, (res['bbox']['x_min'], res['bbox']['y_min'] - text_height - baseline), (res['bbox']['x_min'] + text_width, res['bbox']['y_min']), class_color, -1)
        cv2.putText(image, text, (res['bbox']['x_min'], res['bbox']['y_min'] - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        if 'object_id' in res and centers is not None:
            centers[res['object_id']].append((int((res['bbox']['x_min'] + res['bbox']['x_max']) / 2), int((res['bbox']['y_min'] + res['bbox']['y_max']) / 2)))
            for j in range(1, len(centers[res['object_id']])):
                if centers[res['object_id']][j - 1] is None or centers[res['object_id']][j] is None:
                    continue
                thickness = int(np.sqrt(64 / float(j + 1)) * 2)
                cv2.line(image, centers[res['object_id']][j - 1], centers[res['object_id']][j], class_color, thickness)
    return image

# Function to process images
def image_processing(frame, model, tracker=None, centers=None):
    results = model.predict(frame)
    result_list_json = result_to_json(results[0], tracker=tracker)
    result_image = view_result(results[0], result_list_json, centers=centers)
    object_count = len(result_list_json)
    return result_image, result_list_json, object_count

# Function to process videos
def video_processing(video_file, model, tracker=None, centers=None):
    try:
        results = model.predict(video_file, verbose=False)
        model_name = os.path.basename(model.ckpt_path).split('.')[0]
        video_file_name_out = os.path.join('', f"{os.path.splitext(os.path.basename(video_file))[0]}_{model_name}_output.mp4")
        result_video_json_file = os.path.join('', f"{os.path.splitext(os.path.basename(video_file))[0]}_{model_name}_output.json")
        # Remove existing files if they exist
        if os.path.exists(video_file_name_out):
            os.remove(video_file_name_out)
        if os.path.exists(result_video_json_file):
            os.remove(result_video_json_file)
        
        # Open JSON file for writing results
        with open(result_video_json_file, 'a') as json_file:
            temp_file = 'temp_' + video_file 
            video_writer = cv2.VideoWriter(temp_file, cv2.VideoWriter_fourcc(*'mp4v'), 30, (results[0].orig_img.shape[1], results[0].orig_img.shape[0]))
            json_file.write('[\n')

            # Initialize counter variables
            total_counted = 0
            tracked_boxes = {}  # Dictionary to store tracked boxes

            for i, result in enumerate(tqdm(results)):
                result_list_json = result_to_json(result, tracker=tracker)
                if i > 0:
                    json_file.write(',\n')
                json.dump(result_list_json, json_file, indent=2)

                for box in result_list_json:
                    try:
                        center_x = int((box['bbox']['x_min'] + box['bbox']['x_max']) / 2)
                        center_y = int((box['bbox']['y_min'] + box['bbox']['y_max']) / 2)
                        box_id = box.get('object_id')  # Get object ID if available

                        # Check if box is on the left side and not already tracked
                        if center_x < result.orig_img.shape[1] // 2 and box_id not in tracked_boxes:
                            total_counted += 1
                            tracked_boxes[box_id] = True  # Mark box as tracked to avoid duplicate counting
                    except KeyError as e:
                        print(f"KeyError: {e}")
                        continue

                # Draw the bounding boxes and IDs
                for res in result_list_json:
                    # Draw the bounding box
                    cv2.rectangle(result.orig_img, (int(res['bbox']['x_min']), int(res['bbox']['y_min'])), (int(res['bbox']['x_max']), int(res['bbox']['y_max'])), (0, 0, 255), 2)
                    # Draw the object ID
                    cv2.putText(result.orig_img, f'ID: {res["object_id"]}', (int(res['bbox']['x_min']), int(res['bbox']['y_min']) - 10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)

                cv2.putText(result.orig_img, f'Total Counted: {total_counted}', (10, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                video_writer.write(result.orig_img)
                
                if i < len(results) - 1:
                    json_file.write(',\n')
            json_file.write('\n]')
            video_writer.release()

        ffmpeg_cmd = f'ffmpeg -i "{temp_file}" -c:v libx264 "{video_file_name_out}"'
        subprocess.run(ffmpeg_cmd, shell=True, check=True)
        os.remove(temp_file)
        
        return video_file_name_out, result_video_json_file, total_counted
    
    except Exception as e:
        print("Error occurred:", e)
        return None, None, 0