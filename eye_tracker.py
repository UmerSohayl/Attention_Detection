import sys
import cv2
import mediapipe as mp
import numpy as np
import time
import json
import threading
from datetime import datetime, timedelta
from collections import deque, Counter
from PyQt5.QtCore import Qt, QTimer, QDateTime, QUrl, QThread, pyqtSignal
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QPushButton, QTextEdit,
                             QGroupBox, QStatusBar, QFileDialog, QMessageBox,
                             QCheckBox, QSpinBox, QFormLayout, QProgressBar,
                             QTabWidget, QSlider, QComboBox, QDialog, QGridLayout)
from PyQt5.QtGui import QImage, QPixmap, QPalette, QColor, QTextCursor, QFont
from PyQt5.QtMultimedia import QSoundEffect
from PyQt5.QtChart import QChart, QChartView, QLineSeries, QPieSeries, QValueAxis, QDateTimeAxis
from PyQt5.QtCore import QDateTime as QtDateTime

# Eye and iris landmark indices
LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

class CalibrationDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Eye Tracking Calibration")
        self.setGeometry(200, 200, 400, 300)
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        self.instruction_label = QLabel("Look at the center of the screen and click 'Capture Center'")
        self.instruction_label.setAlignment(Qt.AlignCenter)
        self.instruction_label.setWordWrap(True)
        layout.addWidget(self.instruction_label)
        
        self.progress = QProgressBar()
        self.progress.setMaximum(5)
        layout.addWidget(self.progress)
        
        button_layout = QHBoxLayout()
        self.capture_btn = QPushButton("Capture Center")
        self.capture_btn.clicked.connect(self.capture_position)
        button_layout.addWidget(self.capture_btn)
        
        self.skip_btn = QPushButton("Skip Calibration")
        self.skip_btn.clicked.connect(self.reject)
        button_layout.addWidget(self.skip_btn)
        
        layout.addLayout(button_layout)
        
        self.positions = ['center', 'left', 'right', 'up', 'down']
        self.current_position = 0
        self.calibration_data = {}
        
    def capture_position(self):
        position = self.positions[self.current_position]
        # Signal parent to capture current eye position
        self.parent().capture_calibration_point(position)
        
        self.current_position += 1
        self.progress.setValue(self.current_position)
        
        if self.current_position < len(self.positions):
            next_position = self.positions[self.current_position]
            instructions = {
                'left': 'Look to the LEFT of the screen',
                'right': 'Look to the RIGHT of the screen', 
                'up': 'Look UP',
                'down': 'Look DOWN'
            }
            self.instruction_label.setText(f"{instructions[next_position]} and click 'Capture {next_position.title()}'")
            self.capture_btn.setText(f"Capture {next_position.title()}")
        else:
            self.instruction_label.setText("Calibration Complete!")
            self.capture_btn.setText("Finish")
            self.capture_btn.clicked.disconnect()
            self.capture_btn.clicked.connect(self.accept)

class VideoProcessingThread(QThread):
    frame_processed = pyqtSignal(np.ndarray, dict)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.cap = None
        self.face_mesh = None
        self.running = False
        self.parent_app = parent
        
    def initialize_camera(self):
        self.cap = cv2.VideoCapture(0)
        mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        return self.cap.isOpened()
        
    def run(self):
        self.running = True
        while self.running and self.cap and self.cap.isOpened():
            success, img = self.cap.read()
            if success:
                img = cv2.flip(img, 1)
                data = self.process_frame(img)
                self.frame_processed.emit(img, data)
            time.sleep(0.033)  # ~30 fps
            
    def process_frame(self, img):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, c = img.shape
        results = self.face_mesh.process(imgRGB)
        
        data = {
            'face_detected': False,
            'direction': 'CENTER',
            'confidence': 0.0,
            'eye_openness': 1.0
        }
        
        if results.multi_face_landmarks:
            data['face_detected'] = True
            points = np.array([np.multiply([p.x, p.y], [w, h]).astype(int)
                              for p in results.multi_face_landmarks[0].landmark])
            
            # Calculate iris centers
            left_iris_center = self.calc_center(LEFT_IRIS, points)
            right_iris_center = self.calc_center(RIGHT_IRIS, points)
            
            # Determine gaze direction
            left_direction = self.eye_direction(left_iris_center, points[LEFT_EYE])
            right_direction = self.eye_direction(right_iris_center, points[RIGHT_EYE])
            
            # Combine directions
            data['direction'] = self.combine_directions(left_direction, right_direction)
            data['confidence'] = self.calculate_confidence(points)
            data['eye_openness'] = self.calculate_eye_openness(points)
            
        return data
    
    def calc_center(self, indices, points):
        if len(indices) == 0:
            return np.array([0, 0])
        array = points[indices]
        return np.mean(array, axis=0).astype(int)
    
    def eye_direction(self, iris_center, eye_points):
        if len(eye_points) == 0:
            return "CENTER"
            
        x_min, y_min = np.min(eye_points, axis=0)
        x_max, y_max = np.max(eye_points, axis=0)
        
        eye_width = float(x_max - x_min)
        eye_height = float(y_max - y_min)
        
        if eye_width <= 0 or eye_height <= 0:
            return "CENTER"
            
        x_ratio = (float(iris_center[0]) - x_min) / eye_width
        y_ratio = (float(iris_center[1]) - y_min) / eye_height
        
        # Use calibrated thresholds if available
        h_thresh = getattr(self.parent_app, 'calibrated_h_threshold', 0.35)
        v_thresh = getattr(self.parent_app, 'calibrated_v_threshold', 0.35)
        
        if x_ratio < h_thresh:
            return "LEFT"
        elif x_ratio > (1.0 - h_thresh):
            return "RIGHT"
        elif y_ratio < v_thresh:
            return "UP"
        elif y_ratio > (1.0 - v_thresh):
            return "DOWN"
            
        return "CENTER"
    
    def combine_directions(self, left, right):
        if left == right:
            return left
        elif left == "CENTER":
            return right
        elif right == "CENTER":
            return left
        else:
            # Prefer vertical directions
            if left in ("UP", "DOWN"):
                return left
            elif right in ("UP", "DOWN"):
                return right
            else:
                return left
    
    def calculate_confidence(self, points):
        # Simple confidence based on eye area
        left_eye_area = cv2.contourArea(points[LEFT_EYE]) if len(points[LEFT_EYE]) > 2 else 0
        right_eye_area = cv2.contourArea(points[RIGHT_EYE]) if len(points[RIGHT_EYE]) > 2 else 0
        total_area = left_eye_area + right_eye_area
        return min(1.0, total_area / 1000.0)  # Normalize
    
    def calculate_eye_openness(self, points):
        # Calculate eye aspect ratio for blink detection
        left_ear = self.eye_aspect_ratio(points[LEFT_EYE])
        right_ear = self.eye_aspect_ratio(points[RIGHT_EYE])
        return (left_ear + right_ear) / 2.0
    
    def eye_aspect_ratio(self, eye_points):
        if len(eye_points) < 6:
            return 1.0
        # Vertical distances
        A = np.linalg.norm(eye_points[1] - eye_points[5])
        B = np.linalg.norm(eye_points[2] - eye_points[4])
        # Horizontal distance
        C = np.linalg.norm(eye_points[0] - eye_points[3])
        return (A + B) / (2.0 * C) if C > 0 else 1.0
    
    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()

class FocusAidApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ADHD Focus Aid - Enhanced v2.0")
        self.setGeometry(100, 100, 1600, 1000)
        
        # Initialize variables
        self.is_monitoring = False
        self.direction = "CENTER"
        self.direction_history = deque(maxlen=5)
        self.face_detected = False
        self.t1 = time.time()
        self.warned = False
        self.distraction_start_time = 0
        self.distraction_duration = 0
        self.session_start_time = None
        self.total_focus_time = 0
        self.break_reminder_time = 1800  # 30 minutes
        self.last_focus_check = time.time()
        self.current_focus_streak = 0
        self.longest_focus_streak = 0
        self.focus_streak_start = None
        
        # Configuration
        self.config_file = "focus_config.json"
        self.log_file = "focus_log.txt"
        self.load_config()
        
        # Statistics tracking
        self.session_stats = {
            'total_time': 0,
            'focus_time': 0,
            'distraction_events': 0,
            'directions': Counter(),
            'start_time': None
        }
        
        # Calibration data
        self.calibration_data = {}
        self.calibrated_h_threshold = 0.35
        self.calibrated_v_threshold = 0.35
        
        # Threading
        self.video_thread = VideoProcessingThread(self)
        self.video_thread.frame_processed.connect(self.handle_processed_frame)
        
        # Setup UI
        self.setup_ui()
        
        # Setup timers
        self.ui_timer = QTimer()
        self.ui_timer.timeout.connect(self.update_ui_stats)
        
        # Setup sound
        self.setup_audio()
        
    def load_config(self):
        """Load configuration from file"""
        default_config = {
            'warning_threshold': 2.0,
            'enable_sound': True,
            'show_landmarks': True,
            'break_reminder': 30,
            'sound_volume': 0.5,
            'alert_type': 'beep'
        }
        
        try:
            with open(self.config_file, 'r') as f:
                config = json.load(f)
                self.warning_threshold = config.get('warning_threshold', 2.0)
                self.enable_sound = config.get('enable_sound', True)
                self.show_landmarks = config.get('show_landmarks', True)
                self.break_reminder_time = config.get('break_reminder', 30) * 60
                self.sound_volume = config.get('sound_volume', 0.5)
                self.alert_type = config.get('alert_type', 'beep')
        except FileNotFoundError:
            self.__dict__.update(default_config)
            self.save_config()
            
    def save_config(self):
        """Save configuration to file"""
        config = {
            'warning_threshold': self.warning_threshold,
            'enable_sound': self.enable_sound,
            'show_landmarks': self.show_landmarks,
            'break_reminder': self.break_reminder_time // 60,
            'sound_volume': self.sound_volume,
            'alert_type': self.alert_type
        }
        
        try:
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            print(f"Could not save config: {e}")
            
    def setup_audio(self):
        """Initialize audio system"""
        self.sound_effect = QSoundEffect()
        self.sound_effect.setVolume(self.sound_volume)
        
        # Create a simple beep sound programmatically or use system beep
        self.audio_initialized = False
        try:
            # Try to load custom sound file if it exists
            import os
            if os.path.exists("alert.wav"):
                self.sound_effect.setSource(QUrl.fromLocalFile(os.path.abspath("alert.wav")))
                if self.sound_effect.status() == QSoundEffect.Ready:
                    self.audio_initialized = True
        except Exception as e:
            print(f"Audio setup warning: {e}")
            
        # Fallback: always use system beep
        print(f"Audio initialized: {self.audio_initialized}")
            
    def setup_ui(self):
        """Setup the user interface"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Create tab widget
        tab_widget = QTabWidget()
        central_widget_layout = QVBoxLayout(central_widget)
        central_widget_layout.addWidget(tab_widget)
        
        # Main monitoring tab
        main_tab = self.create_main_tab()
        tab_widget.addTab(main_tab, "Monitor")
        
        # Statistics tab
        stats_tab = self.create_stats_tab()
        tab_widget.addTab(stats_tab, "Statistics")
        
        # Settings tab
        settings_tab = self.create_settings_tab()
        tab_widget.addTab(settings_tab, "Settings")
        
        self.apply_dark_theme()
        self.statusBar().showMessage("Ready - Click Start to begin monitoring")
        
    def create_main_tab(self):
        """Create main monitoring interface"""
        main_widget = QWidget()
        main_layout = QHBoxLayout(main_widget)
        
        # Left panel - video and status
        left_panel = QVBoxLayout()
        
        # Video display
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setText("Camera feed will appear here")
        self.video_label.setStyleSheet("border: 2px solid #555; background-color: #2a2a2a; border-radius: 8px;")
        left_panel.addWidget(self.video_label)
        
        # Status indicators
        status_grid = QGridLayout()
        
        self.status_label = QLabel("Status: Ready")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("font-weight: bold; font-size: 16px; padding: 10px; border-radius: 5px; background-color: #333;")
        status_grid.addWidget(self.status_label, 0, 0, 1, 2)
        
        self.direction_label = QLabel("Direction: CENTER")
        self.direction_label.setAlignment(Qt.AlignCenter)
        self.direction_label.setStyleSheet("font-size: 14px; padding: 8px; border-radius: 5px; background-color: #333;")
        status_grid.addWidget(self.direction_label, 1, 0)
        
        self.confidence_label = QLabel("Confidence: 0%")
        self.confidence_label.setAlignment(Qt.AlignCenter)
        self.confidence_label.setStyleSheet("font-size: 14px; padding: 8px; border-radius: 5px; background-color: #333;")
        status_grid.addWidget(self.confidence_label, 1, 1)
        
        self.face_status_label = QLabel("Face: Not Detected")
        self.face_status_label.setAlignment(Qt.AlignCenter)
        self.face_status_label.setStyleSheet("font-size: 14px; padding: 8px; border-radius: 5px; background-color: #333;")
        status_grid.addWidget(self.face_status_label, 2, 0)
        
        self.distraction_label = QLabel("Current Distraction: 0.0s")
        self.distraction_label.setAlignment(Qt.AlignCenter)
        self.distraction_label.setStyleSheet("font-size: 14px; padding: 8px; border-radius: 5px; background-color: #333;")
        status_grid.addWidget(self.distraction_label, 2, 1)
        
        left_panel.addLayout(status_grid)
        
        # Session info
        session_layout = QHBoxLayout()
        self.session_time_label = QLabel("Session: 00:00:00")
        self.focus_time_label = QLabel("Focus: 00:00:00")
        self.focus_percentage_label = QLabel("Focus: 0%")
        
        for label in [self.session_time_label, self.focus_time_label, self.focus_percentage_label]:
            label.setAlignment(Qt.AlignCenter)
            label.setStyleSheet("font-size: 14px; padding: 8px; border-radius: 5px; background-color: #333; font-weight: bold;")
            session_layout.addWidget(label)
            
        left_panel.addLayout(session_layout)
        
        # Right panel - controls and log
        right_panel = QVBoxLayout()
        
        # Controls
        controls_group = QGroupBox("Controls")
        controls_layout = QVBoxLayout(controls_group)
        
        self.start_btn = QPushButton("Start Monitoring")
        self.start_btn.clicked.connect(self.toggle_monitoring)
        self.start_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; padding: 12px; border-radius: 6px; font-size: 14px; }")
        controls_layout.addWidget(self.start_btn)
        
        self.calibrate_btn = QPushButton("Calibrate Eye Tracking")
        self.calibrate_btn.clicked.connect(self.start_calibration)
        self.calibrate_btn.setStyleSheet("QPushButton { background-color: #2196F3; color: white; padding: 10px; border-radius: 5px; }")
        controls_layout.addWidget(self.calibrate_btn)
        
        self.break_btn = QPushButton("Take Break")
        self.break_btn.clicked.connect(self.take_break)
        self.break_btn.setStyleSheet("QPushButton { background-color: #FF9800; color: white; padding: 10px; border-radius: 5px; }")
        controls_layout.addWidget(self.break_btn)
        
        # Quick settings
        quick_settings = QFormLayout()
        
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setRange(1, 10)
        self.threshold_slider.setValue(int(self.warning_threshold))
        self.threshold_slider.valueChanged.connect(self.update_threshold)
        self.threshold_label = QLabel(f"Warning: {self.warning_threshold}s")
        quick_settings.addRow(self.threshold_label, self.threshold_slider)
        
        controls_layout.addLayout(quick_settings)
        right_panel.addWidget(controls_group)
        
        # Activity log
        log_group = QGroupBox("Activity Log")
        log_layout = QVBoxLayout(log_group)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(300)
        log_layout.addWidget(self.log_text)
        
        log_buttons = QHBoxLayout()
        self.clear_log_btn = QPushButton("Clear")
        self.clear_log_btn.clicked.connect(self.clear_log)
        self.export_btn = QPushButton("Export")
        self.export_btn.clicked.connect(self.export_log)
        
        log_buttons.addWidget(self.clear_log_btn)
        log_buttons.addWidget(self.export_btn)
        log_layout.addLayout(log_buttons)
        
        right_panel.addWidget(log_group)
        
        # Add panels to main layout
        main_layout.addLayout(left_panel, 2)
        main_layout.addLayout(right_panel, 1)
        
        return main_widget
        
    def create_stats_tab(self):
        """Create statistics dashboard"""
        stats_widget = QWidget()
        layout = QVBoxLayout(stats_widget)
        
        # Session summary
        summary_group = QGroupBox("Session Summary")
        summary_layout = QGridLayout(summary_group)
        
        self.stats_labels = {}
        stats_items = [
            ("Total Time", "00:00:00"),
            ("Focus Time", "00:00:00"), 
            ("Distractions", "0"),
            ("Focus Rate", "0%"),
            ("Avg Distraction", "0.0s"),
            ("Longest Focus", "00:00:00")
        ]
        
        for i, (label, value) in enumerate(stats_items):
            row, col = i // 2, (i % 2) * 2
            summary_layout.addWidget(QLabel(label + ":"), row, col)
            stat_label = QLabel(value)
            stat_label.setStyleSheet("font-weight: bold; color: #4CAF50;")
            self.stats_labels[label] = stat_label
            summary_layout.addWidget(stat_label, row, col + 1)
            
        layout.addWidget(summary_group)
        
        # Charts placeholder (would need PyQt5 charts for full implementation)
        charts_group = QGroupBox("Focus Patterns")
        charts_layout = QVBoxLayout(charts_group)
        self.chart_placeholder = QLabel("Charts would appear here\n(Requires PyQt5 Charts extension)")
        self.chart_placeholder.setAlignment(Qt.AlignCenter)
        self.chart_placeholder.setMinimumHeight(200)
        self.chart_placeholder.setStyleSheet("border: 1px dashed #555; background-color: #2a2a2a;")
        charts_layout.addWidget(self.chart_placeholder)
        layout.addWidget(charts_group)
        
        return stats_widget
        
    def create_settings_tab(self):
        """Create settings interface"""
        settings_widget = QWidget()
        layout = QVBoxLayout(settings_widget)
        
        # Detection settings
        detection_group = QGroupBox("Detection Settings")
        detection_layout = QFormLayout(detection_group)
        
        self.threshold_spin = QSpinBox()
        self.threshold_spin.setRange(1, 10)
        self.threshold_spin.setValue(int(self.warning_threshold))
        self.threshold_spin.valueChanged.connect(self.update_threshold)
        detection_layout.addRow("Warning Threshold (s):", self.threshold_spin)
        
        self.landmarks_check = QCheckBox("Show Eye Landmarks")
        self.landmarks_check.setChecked(self.show_landmarks)
        self.landmarks_check.stateChanged.connect(self.toggle_landmarks)
        detection_layout.addRow(self.landmarks_check)
        
        # Audio settings
        audio_group = QGroupBox("Audio Settings")
        audio_layout = QFormLayout(audio_group)
        
        self.sound_check = QCheckBox("Enable Sound Alerts")
        self.sound_check.setChecked(self.enable_sound)
        self.sound_check.stateChanged.connect(self.toggle_sound)
        audio_layout.addRow(self.sound_check)
        
        self.volume_slider = QSlider(Qt.Horizontal)
        self.volume_slider.setRange(0, 100)
        self.volume_slider.setValue(int(self.sound_volume * 100))
        self.volume_slider.valueChanged.connect(self.update_volume)
        audio_layout.addRow("Volume:", self.volume_slider)
        
        self.alert_combo = QComboBox()
        self.alert_combo.addItems(["Beep", "Gentle", "Voice"])
        audio_layout.addRow("Alert Type:", self.alert_combo)
        
        # Break settings
        break_group = QGroupBox("Break Reminders")
        break_layout = QFormLayout(break_group)
        
        self.break_spin = QSpinBox()
        self.break_spin.setRange(5, 120)
        self.break_spin.setValue(self.break_reminder_time // 60)
        self.break_spin.valueChanged.connect(self.update_break_time)
        break_layout.addRow("Remind every (min):", self.break_spin)
        
        # Add all groups
        layout.addWidget(detection_group)
        layout.addWidget(audio_group) 
        layout.addWidget(break_group)
        layout.addStretch()
        
        # Save/Load buttons
        button_layout = QHBoxLayout()
        save_btn = QPushButton("Save Settings")
        save_btn.clicked.connect(self.save_config)
        reset_btn = QPushButton("Reset to Defaults")
        reset_btn.clicked.connect(self.reset_settings)
        
        button_layout.addWidget(save_btn)
        button_layout.addWidget(reset_btn)
        layout.addLayout(button_layout)
        
        return settings_widget
        
    def apply_dark_theme(self):
        """Apply dark theme styling"""
        dark_palette = QPalette()
        dark_palette.setColor(QPalette.Window, QColor(53, 53, 53))
        dark_palette.setColor(QPalette.WindowText, Qt.white)
        dark_palette.setColor(QPalette.Base, QColor(25, 25, 25))
        dark_palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
        dark_palette.setColor(QPalette.ToolTipBase, Qt.white)
        dark_palette.setColor(QPalette.ToolTipText, Qt.white)
        dark_palette.setColor(QPalette.Text, Qt.white)
        dark_palette.setColor(QPalette.Button, QColor(53, 53, 53))
        dark_palette.setColor(QPalette.ButtonText, Qt.white)
        dark_palette.setColor(QPalette.BrightText, Qt.red)
        dark_palette.setColor(QPalette.Link, QColor(42, 130, 218))
        dark_palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
        dark_palette.setColor(QPalette.HighlightedText, Qt.black)
        QApplication.setPalette(dark_palette)
        
        self.setStyleSheet("""
            QMainWindow { background-color: #353535; }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #555;
                border-radius: 8px;
                margin-top: 1ex;
                padding-top: 15px;
                background-color: #404040;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 15px;
                padding: 0 8px 0 8px;
                color: #FFF;
                font-size: 14px;
            }
            QTextEdit {
                background-color: #2a2a2a;
                color: #FFF;
                border: 1px solid #555;
                border-radius: 5px;
                font-family: 'Consolas', monospace;
            }
            QSpinBox, QSlider, QComboBox {
                padding: 6px;
                background-color: #454545;
                color: white;
                border: 1px solid #666;
                border-radius: 4px;
            }
            QTabWidget::pane {
                border: 1px solid #555;
                background-color: #404040;
            }
            QTabBar::tab {
                background-color: #353535;
                color: white;
                padding: 8px 16px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background-color: #404040;
                border-bottom: 2px solid #4CAF50;
            }
        """)
        
    def start_calibration(self):
        """Start eye tracking calibration"""
        if self.is_monitoring:
            QMessageBox.information(self, "Calibration", "Please stop monitoring first.")
            return
            
        dialog = CalibrationDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            self._log("Eye tracking calibration completed", "INFO")
            QMessageBox.information(self, "Calibration", "Calibration completed successfully!")
        
    def capture_calibration_point(self, position):
        """Capture calibration data for a specific gaze position"""
        # This would capture current eye position data
        # For now, just store placeholder data
        self.calibration_data[position] = {
            'timestamp': time.time(),
            'position': position
        }
        
    def toggle_monitoring(self):
        """Start or stop monitoring"""
        if not self.is_monitoring:
            self.start_monitoring()
        else:
            self.stop_monitoring()
            
    def start_monitoring(self):
        """Begin focus monitoring"""
        if not self.video_thread.initialize_camera():
            QMessageBox.critical(self, "Error", "Cannot open camera")
            return
            
        self.is_monitoring = True
        self.session_start_time = time.time()
        self.t1 = time.time()
        self.warned = False
        self.distraction_start_time = 0
        self.distraction_duration = 0
        
        # Reset session statistics
        self.session_stats = {
            'total_time': 0,
            'focus_time': 0,
            'distraction_events': 0,
            'directions': Counter(),
            'start_time': time.time()
        }
        self.total_focus_time = 0
        self.last_focus_check = time.time()
        self.current_focus_streak = 0
        self.longest_focus_streak = 0
        self.focus_streak_start = None
        
        # Update UI
        self.start_btn.setText("Stop Monitoring")
        self.start_btn.setStyleSheet("QPushButton { background-color: #F44336; color: white; font-weight: bold; padding: 12px; border-radius: 6px; font-size: 14px; }")
        self.status_label.setText("Status: Monitoring")
        self.status_label.setStyleSheet("font-weight: bold; color: #4CAF50; font-size: 16px; padding: 10px; border-radius: 5px; background-color: #333;")
        
        # Start threads and timers
        self.video_thread.start()
        self.ui_timer.start(100)  # Update UI every 100ms
        
        self.statusBar().showMessage("Monitoring started - Stay focused!")
        self._log("Focus monitoring session started", "INFO")
        
    def stop_monitoring(self):
        """Stop focus monitoring"""
        self.is_monitoring = False
        
        # Stop threads
        self.video_thread.stop()
        self.video_thread.wait(2000)  # Wait up to 2 seconds
        self.ui_timer.stop()
        
        # Calculate session statistics
        if self.session_start_time:
            total_session_time = time.time() - self.session_start_time
            self.session_stats['total_time'] = total_session_time
            focus_percentage = (self.total_focus_time / total_session_time * 100) if total_session_time > 0 else 0
            self._log(f"Session ended. Duration: {self.format_time(total_session_time)}, Focus: {self.format_time(self.total_focus_time)} ({focus_percentage:.1f}%)", "INFO")
        
        # Update UI
        self.start_btn.setText("Start Monitoring")
        self.start_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; padding: 12px; border-radius: 6px; font-size: 14px; }")
        self.status_label.setText("Status: Stopped")
        self.status_label.setStyleSheet("font-weight: bold; color: #F44336; font-size: 16px; padding: 10px; border-radius: 5px; background-color: #333;")
        
        # Clear video display
        self.video_label.clear()
        self.video_label.setText("Camera feed will appear here")
        self.reset_status_labels()
        
        self.statusBar().showMessage("Monitoring stopped")
        
    def handle_processed_frame(self, img, data):
        """Handle processed frame data from video thread"""
        if not self.is_monitoring:
            return
            
        # Update detection data
        self.face_detected = data['face_detected']
        new_direction = data['direction']
        confidence = data['confidence']
        
        # Apply direction smoothing
        self.direction_history.append(new_direction)
        if len(self.direction_history) >= 3:
            # Use most common direction from recent history
            direction_counts = Counter(self.direction_history)
            self.direction = direction_counts.most_common(1)[0][0]
        else:
            self.direction = new_direction
            
        # Update statistics
        self.session_stats['directions'][self.direction] += 1
        
        # Check for focus/distraction
        self.check_focus_state()
        
        # Update video display
        self.update_video_display(img, data)
        
    def check_focus_state(self):
        """Check current focus state and handle warnings"""
        current_time = time.time()
        time_delta = current_time - self.last_focus_check
        
        # Check if focused (face detected and looking center)
        is_focused = self.face_detected and self.direction == "CENTER"
        
        if is_focused:
            # User is focused - accumulate focus time
            self.total_focus_time += time_delta
            self.session_stats['focus_time'] = self.total_focus_time
            
            # Track focus streak
            if self.focus_streak_start is None:
                self.focus_streak_start = current_time
            self.current_focus_streak = current_time - self.focus_streak_start
            
            # Update longest streak
            if self.current_focus_streak > self.longest_focus_streak:
                self.longest_focus_streak = self.current_focus_streak
            
            if self.warned:
                # Returning to focus
                self.warned = False
                if self.distraction_start_time > 0:
                    self.distraction_duration = current_time - self.distraction_start_time
                    self._log(f"Focus returned after {self.distraction_duration:.1f}s distraction", "INFO")
                    self.distraction_start_time = 0
                    
            self.t1 = current_time
            
        else:
            # User is distracted - reset focus streak
            if self.focus_streak_start is not None:
                self.focus_streak_start = None
                self.current_focus_streak = 0
            
            if not self.warned:
                distraction_time = current_time - self.t1
                if distraction_time > self.warning_threshold:
                    self.trigger_warning()
                    self.warned = True
                    self.distraction_start_time = current_time
                    self.session_stats['distraction_events'] += 1
                    
        self.last_focus_check = current_time
        
        # Check for break reminder
        self.check_break_reminder()
        
    def trigger_warning(self):
        """Trigger distraction warning"""
        reason = "Face not detected" if not self.face_detected else f"Looking {self.direction}"
        self._log(f"Distraction detected: {reason}", "WARN")
        
        # Audio warning - try multiple methods
        if self.enable_sound:
            warning_played = False
            
            try:
                # Method 1: Custom sound effect
                if self.audio_initialized and self.sound_effect.isLoaded():
                    self.sound_effect.play()
                    warning_played = True
                    print("Played custom sound effect")
            except Exception as e:
                print(f"Custom sound failed: {e}")
            
            try:
                # Method 2: System beep (always try this)
                QApplication.beep()
                print("Played system beep")
                warning_played = True
            except Exception as e:
                print(f"System beep failed: {e}")
            
            try:
                # Method 3: Console bell as last resort
                if not warning_played:
                    print("\a")  # Terminal bell
                    print("Used console bell")
            except Exception as e:
                print(f"Console bell failed: {e}")
                
        # Update status
        self.statusBar().showMessage(f"⚠️ Warning: {reason}")
        
        # Visual feedback in log
        self.log_text.append(f"🔔 ALERT: {reason}")
        cursor = self.log_text.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.log_text.setTextCursor(cursor)
        
    def check_break_reminder(self):
        """Check if break reminder should be shown"""
        if not self.session_start_time:
            return
            
        session_time = time.time() - self.session_start_time
        # Check every minute if we've passed a break interval
        if (session_time > self.break_reminder_time and 
            int(session_time) % self.break_reminder_time < 2 and
            not hasattr(self, 'last_break_reminder')):
            self.last_break_reminder = time.time()
            self.suggest_break()
        
        # Reset break reminder flag after some time
        if hasattr(self, 'last_break_reminder') and time.time() - self.last_break_reminder > 60:
            delattr(self, 'last_break_reminder')
            
    def suggest_break(self):
        """Suggest taking a break"""
        msg = QMessageBox(self)
        msg.setWindowTitle("Break Reminder")
        msg.setText("You've been focusing for a while!")
        msg.setInformativeText("Consider taking a short break to rest your eyes and mind.")
        msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Ignore)
        msg.setDefaultButton(QMessageBox.Ok)
        
        if msg.exec_() == QMessageBox.Ok:
            self.take_break()
            
    def take_break(self):
        """Initiate break mode"""
        if self.is_monitoring:
            self.stop_monitoring()
            
        # Simple break timer dialog
        break_dialog = QMessageBox(self)
        break_dialog.setWindowTitle("Break Time")
        break_dialog.setText("Take a 5-minute break!")
        break_dialog.setInformativeText("Look away from the screen, stretch, and relax your eyes.")
        break_dialog.exec_()
        
        self._log("Break taken", "INFO")
        
    def update_video_display(self, img, data):
        """Update the video display with current frame"""
        display_img = img.copy()
        
        # Add warning overlay if needed
        if self.warned:
            overlay = display_img.copy()
            cv2.rectangle(overlay, (0, 0), (display_img.shape[1], display_img.shape[0]), (0, 0, 255), -1)
            cv2.addWeighted(overlay, 0.2, display_img, 0.8, 0, display_img)
            
            cv2.putText(display_img, "FOCUS!", 
                       (display_img.shape[1]//2 - 60, display_img.shape[0]//2),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
                       
        # Add direction indicator
        cv2.putText(display_img, f"Direction: {self.direction}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                   
        # Add confidence indicator
        cv2.putText(display_img, f"Confidence: {data['confidence']:.1%}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                   
        # Convert to Qt format and display
        display_img = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
        h, w, ch = display_img.shape
        bytes_per_line = ch * w
        qt_img = QImage(display_img.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_img)
        
        # Scale to fit display
        self.video_label.setPixmap(pixmap.scaled(
            self.video_label.width(), self.video_label.height(), Qt.KeepAspectRatio))
            
    def update_ui_stats(self):
        """Update UI statistics display"""
        if not self.is_monitoring or not self.session_start_time:
            return
            
        current_time = time.time()
        session_duration = current_time - self.session_start_time
        
        # Update session time
        self.session_time_label.setText(f"Session: {self.format_time(session_duration)}")
        
        # Update focus time
        focus_time = self.total_focus_time  # Use accumulated focus time
        self.focus_time_label.setText(f"Focus: {self.format_time(focus_time)}")
        
        # Update focus percentage
        focus_percentage = (focus_time / session_duration * 100) if session_duration > 0 else 0
        self.focus_percentage_label.setText(f"Focus: {focus_percentage:.1f}%")
        
        # Update other indicators
        self.direction_label.setText(f"Direction: {self.direction}")
        
        if self.face_detected:
            self.face_status_label.setText("Face: Detected")
            self.face_status_label.setStyleSheet("font-size: 14px; padding: 8px; border-radius: 5px; background-color: #2E7D32; color: white;")
        else:
            self.face_status_label.setText("Face: Not Detected")
            self.face_status_label.setStyleSheet("font-size: 14px; padding: 8px; border-radius: 5px; background-color: #D32F2F; color: white;")
            
        # Update current distraction time
        if self.warned and self.distraction_start_time:
            current_distraction = current_time - self.distraction_start_time
            self.distraction_label.setText(f"Current Distraction: {current_distraction:.1f}s")
        else:
            self.distraction_label.setText("Current Distraction: 0.0s")
            
        # Update statistics tab
        self.update_stats_display()
        
    def update_stats_display(self):
        """Update statistics display"""
        if not self.session_start_time:
            return
            
        current_time = time.time()
        total_time = current_time - self.session_start_time
        focus_time = self.total_focus_time  # Use accumulated focus time
        distraction_count = self.session_stats['distraction_events']
        
        # Update statistics labels
        self.stats_labels["Total Time"].setText(self.format_time(total_time))
        self.stats_labels["Focus Time"].setText(self.format_time(focus_time))
        self.stats_labels["Distractions"].setText(str(distraction_count))
        
        focus_rate = (focus_time / total_time * 100) if total_time > 0 else 0
        self.stats_labels["Focus Rate"].setText(f"{focus_rate:.1f}%")
        
        avg_distraction = ((total_time - focus_time) / max(1, distraction_count)) if distraction_count > 0 else 0
        self.stats_labels["Avg Distraction"].setText(f"{avg_distraction:.1f}s")
        
        # Update longest focus streak
        display_longest = max(self.longest_focus_streak, self.current_focus_streak)
        self.stats_labels["Longest Focus"].setText(self.format_time(display_longest))
        
    def format_time(self, seconds):
        """Format time duration as HH:MM:SS"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        
    def reset_status_labels(self):
        """Reset status labels to default state"""
        self.direction_label.setText("Direction: CENTER")
        self.confidence_label.setText("Confidence: 0%")
        self.face_status_label.setText("Face: Not Detected")
        self.face_status_label.setStyleSheet("font-size: 14px; padding: 8px; border-radius: 5px; background-color: #333;")
        self.distraction_label.setText("Current Distraction: 0.0s")
        self.session_time_label.setText("Session: 00:00:00")
        self.focus_time_label.setText("Focus: 00:00:00")
        self.focus_percentage_label.setText("Focus: 0%")
        
    def update_threshold(self, value):
        """Update warning threshold"""
        self.warning_threshold = float(value)
        self.threshold_label.setText(f"Warning: {self.warning_threshold}s")
        
    def update_volume(self, value):
        """Update sound volume"""
        self.sound_volume = value / 100.0
        self.sound_effect.setVolume(self.sound_volume)
        
    def update_break_time(self, value):
        """Update break reminder time"""
        self.break_reminder_time = value * 60
        
    def toggle_sound(self, state):
        """Toggle sound alerts"""
        self.enable_sound = state == Qt.Checked
        
    def toggle_landmarks(self, state):
        """Toggle landmark display"""
        self.show_landmarks = state == Qt.Checked
        
    def reset_settings(self):
        """Reset all settings to defaults"""
        self.warning_threshold = 2.0
        self.enable_sound = True
        self.show_landmarks = True
        self.break_reminder_time = 1800
        self.sound_volume = 0.5
        self.alert_type = 'beep'
        
        # Update UI controls
        self.threshold_spin.setValue(int(self.warning_threshold))
        self.threshold_slider.setValue(int(self.warning_threshold))
        self.sound_check.setChecked(self.enable_sound)
        self.landmarks_check.setChecked(self.show_landmarks)
        self.break_spin.setValue(self.break_reminder_time // 60)
        self.volume_slider.setValue(int(self.sound_volume * 100))
        
        self.save_config()
        QMessageBox.information(self, "Settings", "Settings reset to defaults.")
        
    def clear_log(self):
        """Clear the activity log"""
        self.log_text.clear()
        
    def export_log(self):
        """Export activity log to file"""
        try:
            file_name, _ = QFileDialog.getSaveFileName(
                self, "Export Log", 
                f"focus_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt", 
                "Text Files (*.txt)")
                
            if file_name:
                with open(file_name, 'w') as f:
                    f.write(self.log_text.toPlainText())
                QMessageBox.information(self, "Export", "Log exported successfully!")
                
        except Exception as e:
            QMessageBox.warning(self, "Export Error", f"Could not export log: {e}")
            
    def _log(self, message, level="INFO"):
        """Add entry to activity log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {level}: {message}"
        
        self.log_text.append(log_entry)
        
        # Auto-scroll to bottom
        cursor = self.log_text.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.log_text.setTextCursor(cursor)
        
        # Also write to file
        try:
            with open(self.log_file, 'a') as f:
                full_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"[{full_timestamp}] {level}: {message}\n")
        except Exception:
            pass  # Ignore file write errors
            
    def closeEvent(self, event):
        """Handle application close event"""
        if self.is_monitoring:
            self.stop_monitoring()
            
        # Save final configuration
        self.save_config()
        
        # Clean up
        try:
            self.video_thread.stop()
            self.video_thread.wait(1000)
        except Exception:
            pass
            
        event.accept()


def main():
    """Main application entry point"""
    app = QApplication(sys.argv)
    app.setApplicationName("ADHD Focus Aid - Enhanced")
    app.setApplicationVersion("2.0")
    app.setStyle('Fusion')
    
    # Create and show main window
    window = FocusAidApp()
    window.show()
    
    # Handle application exit
    try:
        sys.exit(app.exec_())
    except SystemExit:
        pass


if __name__ == "__main__":
    main()