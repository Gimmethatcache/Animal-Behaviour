import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFileDialog
from PyQt5.QtGui import QPixmap, QImage, QFont
from PyQt5.QtCore import QTimer, QThread, pyqtSignal, Qt
from PyQt5.QtGui import QMovie
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from keras.models import load_model as load_keras_model
from keras.models import model_from_json
from keras.optimizers import SGD

# Define constants
SEQUENCE_LENGTH = 20
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
CLASSES_LIST = ["CatBiting", "CatEating", "CatSleeping", "DogBiting", "DogDigging"]

# Load the behavior prediction model
print("Loading the behavior prediction model...")
LRCN_model = load_model(r"C:\Users\ksvib\Downloads\Mini Project - 7th Sem\Practical Implementations\Activity_Recognition\LRCN_model___Date_Time_2024_01_30__23_55_29___Loss_1.0159164667129517___Accuracy_0.625.h5")
print("Model loaded successfully.")

# Load the classification model
print("Loading the classification model...")
json_file = open(r"C:/Users/ksvib/Downloads/Mini Project - 7th Sem/Practical Implementations/ResNet 50/model.json", 'r')
model_json_c = json_file.read()
json_file.close()
model_c = model_from_json(model_json_c)
model_c.load_weights(r"C:/Users/ksvib/Downloads/Mini Project - 7th Sem/Practical Implementations/ResNet 50/best_model.h5")
opt = SGD(lr=1e-4, momentum=0.9)
model_c.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])
print("Model loaded successfully.")

# Define the behavior prediction function
def predict_behavior(video_path):
    print("Preprocessing video frames...")
    video_reader = cv2.VideoCapture(video_path)
    video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_list = []
    skip_frames_window = max(int(video_frames_count / SEQUENCE_LENGTH), 1)

    for frame_counter in range(SEQUENCE_LENGTH):
        video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_counter * skip_frames_window)
        success, frame = video_reader.read()
        if not success:
            print("Error: Failed to read frame", frame_counter)
            break
        resized_frame = cv2.resize(frame, (64, 64))
        normalized_frame = resized_frame / 255
        frames_list.append(normalized_frame)

    try:
        print("Performing model prediction...")
        predicted_labels_probabilities = LRCN_model.predict(np.expand_dims(frames_list, axis=0))[0]
        predicted_label_index = np.argmax(predicted_labels_probabilities)
        predicted_behavior = CLASSES_LIST[predicted_label_index]
        print("Predicted behavior:", predicted_behavior)
        return predicted_behavior
    except Exception as e:
        print("An error occurred during behavior prediction:", e)
        return None

# Define the classification function
def predict_on_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Unable to open video file.")
        return None

    predicted_label = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        resized_frame = cv2.resize(frame, (224, 224))
        preds = model_c.predict(np.expand_dims(resized_frame, axis=0))[0]
        if preds[0] < 0.5:
            predicted_label = "Cat"
        else:
            predicted_label = "Dog"

    cap.release()
    cv2.destroyAllWindows()

    return predicted_label

# Define the VideoLabel class for displaying video frames
class VideoLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            q_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.setPixmap(QPixmap.fromImage(q_img))
        else:
            self.timer.stop()

    def set_video(self, video_path):
        self.cap = cv2.VideoCapture(video_path)
        if self.cap.isOpened():
            self.timer.start(1000 // int(self.cap.get(cv2.CAP_PROP_FPS)))
        else:
            print("Error: Unable to open video file.")

# Define the PredictionThread class for performing behavior prediction and classification in separate threads
class PredictionThread(QThread):
    behavior_result_signal = pyqtSignal(str)
    classification_result_signal = pyqtSignal(str)

    def __init__(self, video_path):
        super().__init__()
        self.video_path = video_path

    def run(self):
        predicted_behavior = predict_behavior(self.video_path)
        self.behavior_result_signal.emit(predicted_behavior)

        predicted_label = predict_on_video(self.video_path)
        self.classification_result_signal.emit(predicted_label)

# Define the ModelDeployer class for the GUI
class ModelDeployer(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Custom Model Deployer")
        layout = QVBoxLayout()

        self.upload_button = QPushButton("Upload Video")
        self.upload_button.clicked.connect(self.upload_video)
        layout.addWidget(self.upload_button)

        # Horizontal layout for video preview and loading GIF
        horizontal_layout = QHBoxLayout()

        # Video preview label
        self.video_label = VideoLabel()
        horizontal_layout.addWidget(self.video_label)

        # Loading GIF label
        self.loading_label = QLabel()
        movie = QMovie(r"C:\Users\ksvib\Downloads\Pycharm\PyQt_Behaviour_Prediction\cat_loading.gif")
        self.loading_label.setMovie(movie)
        movie.start()
        horizontal_layout.addWidget(self.loading_label)
        self.loading_label.hide()

        layout.addLayout(horizontal_layout)

        self.predict_button = QPushButton("Predict")
        self.predict_button.clicked.connect(self.predict)
        layout.addWidget(self.predict_button)

        self.result_label = QLabel()
        layout.addWidget(self.result_label)

        self.setLayout(layout)

    def upload_video(self):
        video_file, _ = QFileDialog.getOpenFileName(self, "Open Video", "", "Video files (*.mp4 *.avi)")
        if video_file:
            self.video_path = video_file
            self.video_label.set_video(video_file)

    def predict(self):
        if not hasattr(self, 'video_path'):
            self.result_label.setText("Please upload a video first.")
            return

        self.predict_button.setEnabled(False)
        self.result_label.setText("Predicting... Please wait.")
        self.loading_label.show()  # Show loading GIF

        self.thread = PredictionThread(self.video_path)
        self.thread.behavior_result_signal.connect(self.display_behavior_prediction)
        self.thread.classification_result_signal.connect(self.display_classification)
        self.thread.finished.connect(self.on_thread_finished)
        self.thread.start()

    def display_behavior_prediction(self, predicted_behavior):
        self.behavior_prediction = predicted_behavior

    def display_classification(self, predicted_label):
        self.classification = predicted_label
        self.result_label.setText(f"Predicted Label: {self.classification}, Predicted Behavior: {self.behavior_prediction}")
        self.loading_label.hide()  # Hide loading GIF
        self.predict_button.setEnabled(True)

    def on_thread_finished(self):
        pass

if __name__ == "__main__":
    app = QApplication(sys.argv)
    deployer = ModelDeployer()
    deployer.show()
    sys.exit(app.exec_())
