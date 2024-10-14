import cv2
import torch
from ultralytics import YOLO
from time import time
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage  # Import for attaching images
from ultralytics.utils.plotting import Annotator, colors
from email_settings import from_email, password, to_email
import os

# Email server setup
server = smtplib.SMTP("smtp.gmail.com: 587")
server.starttls()
server.login(from_email, password)

# Directory to save detected images
save_dir = "detected_images"
os.makedirs(save_dir, exist_ok=True)

def send_email(to_email, from_email, object_detected=1, image_path=None):
    """Sends an email notification indicating the number of objects detected; defaults to 1 object."""
    message = MIMEMultipart()
    message["From"] = from_email
    message["To"] = to_email
    message["Subject"] = "Security Alert"
    
    # Add in the message body
    message_body = f"ALERT - {object_detected} object{'s' if object_detected > 1 else ''} has been detected!!"
    message.attach(MIMEText(message_body, "plain"))

    # Attach the detected image if provided
    if image_path:
        with open(image_path, 'rb') as img_file:
            img = MIMEImage(img_file.read())
            img.add_header('Content-Disposition', f'attachment; filename="{os.path.basename(image_path)}"')
            message.attach(img)

    server.sendmail(from_email, to_email, message.as_string())

def detected_image(frame):
    """Saves the detected image in the specified directory and returns its path."""
    timestamp = int(time())
    image_path = os.path.join(save_dir, f"detected_person_{timestamp}.jpg")
    cv2.imwrite(image_path, frame)
    print(f"Saved detected image: {image_path}")
    return image_path  # Return the path for emailing

class ObjectDetection:
    def __init__(self, capture_index):
        """Initializes an ObjectDetection instance with a given camera index."""
        self.capture_index = capture_index
        self.email_sent = False
        self.image_counter = 0  # Counter for sent images

        # Model information
        self.model = YOLO("yolo11n.pt")

        # Visual information
        self.annotator = None

        # Device information
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    # ... (rest of your existing ObjectDetection code)

    def predict(self, im0):
        """Run prediction using a YOLO model for the input image `im0`."""
        results = self.model(im0)
        return results

    def display_fps(self, im0):
        """Displays the FPS on an image `im0` by calculating and overlaying as white text on a black rectangle."""
        end_time = time()
        fps = 1 / round(end_time - self.start_time, 2)
        text = f"FPS: {int(fps)}"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
        gap = 10
        cv2.rectangle(
            im0,
            (20 - gap, 70 - text_size[1] - gap),
            (20 + text_size[0] + gap, 70 + gap),
            (255, 255, 255),
            -1,
        )
        cv2.putText(im0, text, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)

    def plot_bboxes(self, results, im0):
        """Plots bounding boxes on an image given detection results; returns annotated image and class IDs."""
        class_ids = []
        self.annotator = Annotator(im0, 3, results[0].names)
        boxes = results[0].boxes.xyxy.cpu()
        clss = results[0].boxes.cls.cpu().tolist()
        names = results[0].names
        
        for box, cls in zip(boxes, clss):
            class_ids.append(cls)
            self.annotator.box_label(box, label=names[int(cls)], color=colors(int(cls), True))
            
            # Save image if a person is detected (assuming class ID for person is 0)
            if cls == 0:  
                img_path = detected_image(im0)  # Save the current frame when a person is detected

                # Calculate center point of the bounding box for tracking
                x_center = int((box[0] + box[2]) / 2)
                y_center = int((box[1] + box[3]) / 2)

                # Draw a point on the center of the detected person
                cv2.circle(im0, (x_center, y_center), radius=5, color=(255, 0, 0), thickness=-1)  

            if self.image_counter < 5:
                
                send_email(to_email, from_email, len(class_ids), img_path)
                self.image_counter += 1  # Send email with attached image
                
        return im0, class_ids

    def __call__(self):
        """Run object detection on video frames from a camera stream, plotting and showing the results."""
        cap = cv2.VideoCapture(self.capture_index)
        
        assert cap.isOpened()
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        while True:
            self.start_time = time()
            ret, im0 = cap.read()
            assert ret
            
            results = self.predict(im0)
            im0, class_ids = self.plot_bboxes(results, im0)

            if len(class_ids) > 0:  
                if not self.email_sent:
                    send_email(to_email, from_email, len(class_ids))
                    self.email_sent = True
            else:
                self.email_sent = False

            self.display_fps(im0)
            cv2.imshow("YOLO11 Detection", im0)

            if cv2.waitKey(5) & 0xFF == 27:  
                break
        
        cap.release()
        cv2.destroyAllWindows()
        server.quit()

if __name__ == "__main__":
    detector = ObjectDetection(capture_index=0)
    detector()