import torch

import cv2
from PIL import Image

import heapq

# 5. 繪製只標註 "person" 的結果
def plot_results(frame, detections, priority_queue):
    i=0
    for _, row in detections.iterrows():
        # 提取邊界框與置信度
        i=i+1
        x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        conf = row['confidence']
        label = f"{i }Person {conf:.2f}"
        # 繪製邊界框
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # 繪製頭部
        point = (x1 + x2) / 2, (y1 * 9 + y2 * 1) / 10
        cv2.circle(frame, (int(point[0]), int(point[1])), 5, (0, 0, 255), -1)

        heapq.heappush(priority_queue, (-((x2-x1)*(y2-y1)), f"{i}"))

        # 添加標籤文字
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    # 顯示結果圖片
    cv2.imshow("Detected Person", frame)



# 1. 載入 YOLOv5 模型 (使用 COCO 預訓練模型)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# 2. 加載圖片
# image_path = './test.jpg'  # 圖片文件名
# img = Image.open(image_path)

# 2. 加載視訊
video_path = './test.mp4'  # 視訊文件名
cap = cv2.VideoCapture(video_path)

# 獲取視訊基本資訊
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

while True:
    
    ret, frame = cap.read()
    if not ret:
        print("影片播放結束")
        cap.release()
        cv2.destroyAllWindows()
        break
    #3. 使用模型進行檢測
    results = model(frame)

    # 4. 過濾檢測結果 (只保留 "person" 類別)
    detections = results.pandas().xyxy[0]  # 取得檢測結果
    person_detections = detections[detections['name'] == 'person']  # 僅保留 "person"

    priority_queue = []

    #調用函數，只標註 "person"
    plot_results(frame, person_detections, priority_queue)
    
    
    
    if cv2.waitKey(int((1/fps)*1000)) & 0xFF == ord('q'):  # 按 'q' 鍵退出
        break




while priority_queue:
    priority, task = heapq.heappop(priority_queue)
    print(f"Priority: {-priority}, Task: {task}")