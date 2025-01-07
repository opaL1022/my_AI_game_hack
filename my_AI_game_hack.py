import torch
import cv2
from PIL import Image
import heapq

# 1. 載入 YOLOv5 模型 (使用 COCO 預訓練模型)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# 2. 加載圖片
image_path = './test.jpg'  # 圖片文件名
img = Image.open(image_path)

# 3. 使用模型進行檢測
results = model(img)

# 4. 過濾檢測結果 (只保留 "person" 類別)
detections = results.pandas().xyxy[0]  # 取得檢測結果
person_detections = detections[detections['name'] == 'person']  # 僅保留 "person"

priority_queue = []

# 5. 繪製只標註 "person" 的結果
def plot_results(image_path, detections,priority_queue):
    # 讀取圖片
    image = cv2.imread(image_path)
    i=0
    if image is None:
        raise ValueError(f"圖片無法加載，請檢查路徑: {image_path}")
    for _, row in detections.iterrows():
        # 提取邊界框與置信度
        i=i+1
        x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        conf = row['confidence']
        label = f"{i }Person {conf:.2f}"
        # 繪製邊界框
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        heapq.heappush(priority_queue, (-((x2-x1)*(y2-y1)), f"{i}"))

        # 添加標籤文字
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    # 顯示結果圖片
    cv2.imshow("Detected Person", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 調用函數，只標註 "person"
plot_results(image_path, person_detections,priority_queue)

while priority_queue:
    priority, task = heapq.heappop(priority_queue)
    print(f"Priority: {-priority}, Task: {task}")