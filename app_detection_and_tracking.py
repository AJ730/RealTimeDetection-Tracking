import cv2
import numpy as np
from ultralytics import YOLO
import mss
import torch
import threading
from queue import Queue

# Load YOLOv8
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = YOLO("yolov8n.pt").to(device)

# Define classes of interest
classes_of_interest = [
    "person", "bicycle", "car", "motorcycle", "bus", "truck", "traffic light"]

# Model class
class YOLOModel:
    def __init__(self, model_path, device):
        self.device = device
        self.model = YOLO(model_path).to(device)
        self.classes_of_interest = classes_of_interest

    def detect_objects(self, image):
        results = self.model(image)
        return results

# View class
class ScreenCaptureView:
    def __init__(self):
        self.roi = None
        self.drawing = False
        self.resizing = False
        self.dragging = False
        self.start_point = None
        self.end_point = None
        self.selected_corner = None
        self.corners = []
        self.drag_start_point = None
        self.tracker = None
        self.tracking = False
        self.tracking_roi = None
        self.screen_img = None
        self.click_x, self.click_y = None, None

    def set_mouse_callback(self, window_name, callback):
        cv2.setMouseCallback(window_name, callback)

    def update_screen_image(self, screen_img):
        self.screen_img = screen_img

    def draw_rectangle(self, event, x, y, flags, param):
        screen_width, screen_height = param
        x = clamp(x, 0, screen_width - 1)
        y = clamp(y, 0, screen_height - 1)

        if event == cv2.EVENT_LBUTTONDOWN:
            if not self.drawing and not self.resizing and not self.dragging:
                if self.roi and any([abs(x - cx) < 10 and abs(y - cy) < 10 for cx, cy in self.corners]):
                    self.resizing = True
                    self.selected_corner = next((i for i, (cx, cy) in enumerate(self.corners) if abs(x - cx) < 10 and abs(y - cy) < 10), None)
                elif self.roi and (self.roi[0] < x < self.roi[2]) and (self.roi[1] < y < self.roi[3]):
                    self.dragging = True
                    self.drag_start_point = (x, y)
                else:
                    self.drawing = True
                    self.start_point = (x, y)
                    self.end_point = (x, y)

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.end_point = (x, y)
            elif self.resizing and self.selected_corner is not None:
                x1, y1, x2, y2 = self.roi
                if self.selected_corner == 0:
                    self.start_point = (x, y)
                elif self.selected_corner == 1:
                    self.start_point = (x1, y)
                    self.end_point = (x, y2)
                elif self.selected_corner == 2:
                    self.end_point = (x, y)
                elif self.selected_corner == 3:
                    self.start_point = (x, y1)
                    self.end_point = (x2, y)
                self.roi = (clamp(self.start_point[0], 0, screen_width - 1), clamp(self.start_point[1], 0, screen_height - 1),
                            clamp(self.end_point[0], 0, screen_width - 1), clamp(self.end_point[1], 0, screen_height - 1))
                self.corners = [(self.roi[0], self.roi[1]), (self.roi[2], self.roi[1]), (self.roi[2], self.roi[3]), (self.roi[0], self.roi[3])]
            elif self.dragging and self.drag_start_point:
                dx, dy = x - self.drag_start_point[0], y - self.drag_start_point[1]
                x1, y1, x2, y2 = self.roi
                self.roi = (clamp(x1 + dx, 0, screen_width - 1), clamp(y1 + dy, 0, screen_height - 1),
                            clamp(x2 + dx, 0, screen_width - 1), clamp(y2 + dy, 0, screen_height - 1))
                self.drag_start_point = (x, y)
                self.corners = [(self.roi[0], self.roi[1]), (self.roi[2], self.roi[1]), (self.roi[2], self.roi[3]), (self.roi[0], self.roi[3])]

        elif event == cv2.EVENT_LBUTTONUP:
            if self.drawing:
                self.drawing = False
                if self.start_point != self.end_point:
                    self.roi = (clamp(self.start_point[0], 0, screen_width - 1), clamp(self.start_point[1], 0, screen_height - 1),
                                clamp(self.end_point[0], 0, screen_width - 1), clamp(self.end_point[1], 0, screen_height - 1))
                    self.corners = [(self.roi[0], self.roi[1]), (self.roi[2], self.roi[1]), (self.roi[2], self.roi[3]), (self.roi[0], self.roi[3])]
                else:
                    self.roi = None
            elif self.resizing:
                self.resizing = False
                if self.start_point != self.end_point:
                    self.roi = (clamp(self.start_point[0], 0, screen_width - 1), clamp(self.start_point[1], 0, screen_height - 1),
                                clamp(self.end_point[0], 0, screen_width - 1), clamp(self.end_point[1], 0, screen_height - 1))
                    self.corners = [(self.roi[0], self.roi[1]), (self.roi[2], self.roi[1]), (self.roi[2], self.roi[3]), (self.roi[0], self.roi[3])]
                else:
                    self.roi = None
                self.selected_corner = None
            elif self.dragging:
                self.dragging = False
                self.drag_start_point = None

        elif event == cv2.EVENT_RBUTTONDOWN:
            self.click_x, self.click_y = x, y

        elif event == cv2.EVENT_RBUTTONDBLCLK:
            self.tracking = False
            self.tracker = None
            self.tracking_roi = None

    def display(self, window_name):
        cv2.imshow(window_name, self.screen_img)

# Controller class
class Controller:
    def __init__(self, model, view):
        self.model = model
        self.view = view
        self.frame_queue = Queue(maxsize=1)
        self.result_queue = Queue(maxsize=1)
        self.monitor = None
        self.screen_width = None
        self.screen_height = None

    def start(self):
        with mss.mss() as sct:
            self.monitor = sct.monitors[1]
            self.screen_width = self.monitor["width"]
            self.screen_height = self.monitor["height"]

            cv2.namedWindow('Live Feed')
            self.view.set_mouse_callback('Live Feed', lambda event, x, y, flags, param: self.view.draw_rectangle(event, x, y, flags, (self.screen_width, self.screen_height)))

            processing_thread = threading.Thread(target=self.process_frame_worker)
            processing_thread.start()

            while True:
                screen_img = np.array(sct.grab(self.monitor))
                screen_img = cv2.cvtColor(screen_img, cv2.COLOR_BGRA2BGR)
                self.view.update_screen_image(screen_img)

                if self.view.drawing and self.view.start_point and self.view.end_point:
                    img_copy = screen_img.copy()
                    cv2.rectangle(img_copy, self.view.start_point, self.view.end_point, (0, 0, 255), 2)
                    cv2.imshow('Live Feed', img_copy)
                else:
                    if self.view.roi:
                        x1, y1, x2, y2 = self.view.roi
                        cv2.rectangle(screen_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        for cx, cy in self.view.corners:
                            cv2.circle(screen_img, (cx, cy), 5, (0, 0, 255), -1)

                        if not self.frame_queue.full() and (x2 - x1 > 10) and (y2 - y1 > 10):
                            self.frame_queue.put((screen_img, self.view.roi))

                        if not self.result_queue.empty():
                            results, roi_coords = self.result_queue.get()
                            self.process_frame(results, screen_img, roi_coords[0], roi_coords[1])

                    if self.view.tracking and self.view.tracker is not None:
                        success, self.view.tracking_roi = self.view.tracker.update(screen_img)
                        if success:
                            p1 = (int(self.view.tracking_roi[0]), int(self.view.tracking_roi[1]))
                            p2 = (int(self.view.tracking_roi[0] + self.view.tracking_roi[2]), int(self.view.tracking_roi[1] + self.view.tracking_roi[3]))
                            if self.view.roi and (p1[0] < self.view.roi[0] or p1[1] < self.view.roi[1] or p2[0] > self.view.roi[2] or p2[1] > self.view.roi[3]):
                                self.view.tracking = False
                                self.view.tracker = None
                                self.view.tracking_roi = None
                            else:
                                cv2.rectangle(screen_img, p1, p2, (255, 0, 0), 2)
                        else:
                            self.view.tracking = False
                            self.view.tracker = None
                            self.view.tracking_roi = None

                    self.view.display('Live Feed')

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            self.frame_queue.put(None)
            processing_thread.join()
            cv2.destroyAllWindows()

    def process_frame_worker(self):
        while True:
            frame_data = self.frame_queue.get()
            if frame_data is None:
                break
            frame, roi_coords = frame_data
            x1, y1, x2, y2 = roi_coords
            roi_img = frame[y1:y2, x1:x2]

            results = self.model.detect_objects(roi_img)

            self.result_queue.put((results, roi_coords))

    def process_frame(self, results, screen_img, x_offset, y_offset):
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)

            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes[i]
                conf = confidences[i]
                cls = class_ids[i]
                label = self.model.model.names[cls]
                if label in self.model.classes_of_interest and conf > 0.5:
                    if not self.view.tracking:
                        color = (0, 255, 0)
                        cv2.rectangle(screen_img, (int(x1) + x_offset, int(y1) + y_offset), (int(x2) + x_offset, int(y2) + y_offset), color, 2)
                        cv2.putText(screen_img, label, (int(x1) + x_offset, int(y1) + y_offset - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                    if self.view.click_x is not None and self.view.click_y is not None and not self.view.tracking:
                        if int(x1) + x_offset <= self.view.click_x <= int(x2) + x_offset and int(y1) + y_offset <= self.view.click_y <= int(y2) + y_offset:
                            self.view.tracker = cv2.TrackerCSRT_create()
                            self.view.tracking_roi = (int(x1) + x_offset, int(y1) + y_offset, int(x2) - int(x1), int(y2) - int(y1))
                            self.view.tracker.init(screen_img, self.view.tracking_roi)
                            self.view.tracking = True
                            self.view.click_x, self.view.click_y = None, None

def clamp(value, min_value, max_value):
    return max(min(value, max_value), min_value)

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = YOLOModel("yolov8n.pt", device)
    view = ScreenCaptureView()
    controller = Controller(model, view)
    controller.start()
