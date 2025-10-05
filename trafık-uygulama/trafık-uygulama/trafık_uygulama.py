
import cv2
import numpy as np
from ultralytics import YOLO
import time

class TrafficLightSystem:
    def __init__(self):
        self.model = YOLO('yolov8n.pt')
        self.light_color = (0, 0, 255)  # Start with Red
        self.last_update_time = time.time()
        self.check_interval = 5  # Check every 5 seconds
        self.yellow_duration = 3  # Duration to stay in yellow
        self.vehicle_counts = []  # Store vehicle counts for interval
        self.current_status = "Red"
        self.previous_status = "Red"
        self.yellow_start_time = None

    def detect_vehicles(self, frame):
        results = self.model(frame)
        vehicles = results[0].boxes.data.cpu().numpy()
        valid_classes = [2, 3, 5, 7]  # Car, Truck, Bus, Motorcycle
        
        count = 0
        for vehicle in vehicles:
            if int(vehicle[5]) in valid_classes:
                x1, y1, x2, y2 = map(int, vehicle[:4])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, "Vehicle", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                count += 1
        return count

    def update_traffic_light(self, avg_count):
        # Handle yellow delay transition
        if self.current_status == "Yellow":
            if time.time() - self.yellow_start_time >= self.yellow_duration:
                if self.previous_status == "Red":
                    self.current_status = "Green"
                    self.light_color = (0, 255, 0)
                else:
                    self.current_status = "Red"
                    self.light_color = (0, 0, 255)
        else:
            # Determine status based on vehicle average
            if avg_count <= 4 and self.current_status != "Red":
                self.previous_status = self.current_status
                self.current_status = "Yellow"
                self.light_color = (0, 255, 255)
                self.yellow_start_time = time.time()
            elif 5 <= avg_count <= 7 and self.current_status != "Yellow":
                self.previous_status = self.current_status
                self.current_status = "Yellow"
                self.light_color = (0, 255, 255)
                self.yellow_start_time = time.time()
            elif avg_count > 7 and self.current_status != "Green":
                self.previous_status = self.current_status
                self.current_status = "Yellow"
                self.light_color = (0, 255, 255)
                self.yellow_start_time = time.time()

    def draw_traffic_light(self, frame, avg_count):
        # Draw traffic light box
        cv2.rectangle(frame, (10, 10), (110, 210), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (110, 210), (255, 255, 255), 2)

        # Draw lights
        cv2.circle(frame, (60, 50), 30, (0, 0, 255) if self.light_color == (0, 0, 255) else (50, 50, 50), -1)
        cv2.circle(frame, (60, 110), 30, (0, 255, 255) if self.light_color == (0, 255, 255) else (50, 50, 50), -1)
        cv2.circle(frame, (60, 170), 30, (0, 255, 0) if self.light_color == (0, 255, 0) else (50, 50, 50), -1)

        # Show average and current status
        cv2.putText(frame, f"Avg Vehicles: {avg_count:.1f}", (15, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Status: {self.current_status}", (15, 270),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    def start(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Count vehicles in the current frame
                count = self.detect_vehicles(frame)
                self.vehicle_counts.append(count)

                current_time = time.time()

                # Every 5 seconds, calculate average and update light
                if current_time - self.last_update_time >= self.check_interval:
                    if self.vehicle_counts:
                        avg_count = np.mean(self.vehicle_counts)
                        self.update_traffic_light(avg_count)
                        self.vehicle_counts = []  # Reset counts
                    self.last_update_time = current_time

                # Continuously check yellow timeout
                if self.current_status == "Yellow":
                    self.update_traffic_light(0)

                # Draw UI
                avg_count_display = np.mean(self.vehicle_counts) if self.vehicle_counts else 0
                self.draw_traffic_light(frame, avg_count_display)
                cv2.imshow('Traffic Light System', frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                time.sleep(0.1)

        finally:
            cap.release()
            cv2.destroyAllWindows()

print("Starting Traffic Light System...")
system = TrafficLightSystem()
system.start()

