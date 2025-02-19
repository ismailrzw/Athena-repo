from ultralytics import YOLO
import cv2
import os

# Load YOLO model with tracking support
model = YOLO('yolov8s.pt')

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

# Define output video parameters
save_location = "C:/Users/lenovo/Documents"
os.makedirs(save_location, exist_ok=True)
output_filename = os.path.join(save_location, "output.mp4")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_filename, fourcc, 20.0, (640, 480))

# Define a virtual line for counting
line_x = 200
count = 0

# Dictionary to track objects and their movement direction
tracked_objects = {}  # {id: {'prev_x': int, 'direction': str}}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO tracking model on frame
    results = model.track(frame, persist=True)

    # Loop through detected objects
    if results[0].boxes.id is not None:
        for box, obj_id in zip(results[0].boxes.data, results[0].boxes.id):
            x1, y1, x2, y2 = map(int, box[:4])
            obj_id = int(obj_id)

            # Determine object center
            obj_center_x = (x1 + x2) // 2

            # Initialize tracking state for new objects
            if obj_id not in tracked_objects:
                tracked_objects[obj_id] = {'prev_x': obj_center_x, 'direction': None}

            prev_x = tracked_objects[obj_id]['prev_x']

            # Determine movement direction
            if obj_center_x > prev_x:
                tracked_objects[obj_id]['direction'] = "right"
            elif obj_center_x < prev_x:
                tracked_objects[obj_id]['direction'] = "left"

            # Check if the object crosses the line from either side
            if prev_x < line_x and obj_center_x >= line_x and tracked_objects[obj_id]['direction'] == "right":
                count += 1
                print(f"Object {obj_id} crossed right. Count: {count}")

            elif prev_x > line_x and obj_center_x <= line_x and tracked_objects[obj_id]['direction'] == "left":
                count += 1
                print(f"Object {obj_id} crossed left. Count: {count}")

            # Update object previous position
            tracked_objects[obj_id]['prev_x'] = obj_center_x

            # Draw bounding box and label
            label = f'ID:{obj_id}'
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Draw the virtual line
    cv2.line(frame, (line_x, 0), (line_x, frame.shape[0]), (0, 0, 255), 2)

    # Write frame to video and display
    out.write(frame)
    cv2.imshow('YOLO Tracking', frame)

    # Exit on 'q' press or window close
    if cv2.waitKey(1) & 0xFF == ord('q') or cv2.getWindowProperty('YOLO Tracking', cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
out.release()
cv2.destroyAllWindows()
