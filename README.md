# Smart Track Ware: Advanced Inventory Management System.

## Project Overview

Efficient inventory management is vital for streamlined warehouse operations and timely order fulfillment. Traditional manual methods, however, are often inefficient, error-prone, and labor-intensive, leading to delays, inaccuracies, and increased costs. This project introduces an advanced inventory management system utilizing computer vision, a branch of artificial intelligence that enables computers to interpret visual data. The primary objective is to automate inventory management, enhancing efficiency and reducing errors.

## Dataset

Using a dataset of over 9,000 images of cardboard boxes from [Roboflow](https://roboflow.com/), we trained the YOLOv8 object detection model to detect, track, and count boxes in various settings, orientations, and lighting conditions. By leveraging YOLOv8's robust detection capabilities, we aim to achieve high accuracy in inventory tracking.

## Model and Training

The YOLOv8 (You Only Look Once) model was chosen for its state-of-the-art performance in object detection. It is designed to be fast and accurate, making it ideal for real-time applications like inventory management.

### Steps Involved:
1. **Data Preprocessing**: Images were labeled and split into training, validation, and test sets.
2. **Model Training**: The YOLOv8 model was trained on the preprocessed dataset.
3. **Evaluation**: The model's performance was evaluated using metrics such as accuracy, precision, and recall.
![Training Batch](images/train_batch1.jpg)

## Results

The final results demonstrate that our system achieves a high accuracy of 95% in detecting and counting cardboard boxes, leading to more reliable inventory records. This automation allows warehouse staff to focus on strategic tasks, enhancing overall productivity.

## Benefits

- **Increased Efficiency**: Automating the inventory process saves time and reduces labor costs.
- **Reduced Errors**: Accurate detection and counting minimize discrepancies in inventory records.
- **Enhanced Productivity**: Warehouse staff can focus on higher-level tasks, improving overall workflow.
- **Scalability**: The system can be easily scaled to accommodate different warehouse sizes and types of inventory.

## Conclusion

This project underscores the potential of advanced technologies to transform traditional warehouse operations, boosting productivity and customer satisfaction. By integrating computer vision into inventory management, businesses can achieve more efficient and reliable operations.




