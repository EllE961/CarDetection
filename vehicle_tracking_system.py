import cv2
import numpy as np
from shapely.geometry import Point, Polygon
from ultralytics import YOLO
import json
import os
import time
import torch
import argparse  # Added for CLI arguments

#############################
# 1. CONFIG
#############################



# -----------------------
# Checkpoint helpers
# -----------------------

def save_checkpoint(next_video_index, last_frame_number, video_count):
    """Save processing state to allow resuming later (global helper)"""
    global vehicle_paths, last_known_zone, zone_transitions_count

    checkpoint_data = {
        "next_video_index": next_video_index,
        "last_frame_number": last_frame_number,
        "video_count": video_count,
        "vehicle_paths": {str(k): v for k, v in vehicle_paths.items()},
        "last_known_zone": {str(k): v for k, v in last_known_zone.items()},
        "zone_transitions_count": {f"{k[0]}->{k[1]}": v for k, v in zone_transitions_count.items()},
    }

    # Use a temporary file approach to avoid corruption if the program crashes while writing
    temp_checkpoint_file = CHECKPOINT_FILE + ".tmp"
    try:
        # First write to a temporary file
        with open(temp_checkpoint_file, "w") as f:
            json.dump(checkpoint_data, f, indent=4)

        # Then rename the temp file to the actual checkpoint file (atomic operation)
        if os.path.exists(CHECKPOINT_FILE):
            os.replace(temp_checkpoint_file, CHECKPOINT_FILE)  # Overwrite existing file
        else:
            os.rename(temp_checkpoint_file, CHECKPOINT_FILE)  # Create new file

        print(f"Checkpoint saved to {CHECKPOINT_FILE}")
    except Exception as e:
        print(f"Failed to save checkpoint: {e}")
        # Clean up temp file if it exists
        if os.path.exists(temp_checkpoint_file):
            try:
                os.remove(temp_checkpoint_file)
            except Exception:
                pass


def load_checkpoint():
    """Load processing state from checkpoint if available"""
    if not os.path.exists(CHECKPOINT_FILE):
        return None
    try:
        with open(CHECKPOINT_FILE, "r") as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Failed to load checkpoint: {e}")
        return None


def restore_from_checkpoint(checkpoint_data):
    """Restore global tracking state from checkpoint data"""
    global vehicle_paths, last_known_zone, zone_transitions_count
    
    # Restore tracking data
    vehicle_paths = {int(k) if k.isdigit() else k: v for k, v in checkpoint_data["vehicle_paths"].items()}
    last_known_zone = {int(k) if k.isdigit() else k: v for k, v in checkpoint_data["last_known_zone"].items()}
    
    # Restore transition counts (need to parse keys back to tuples)
    zone_transitions = {}
    for k, v in checkpoint_data["zone_transitions_count"].items():
        from_zone, to_zone = k.split("->")
        zone_transitions[(from_zone, to_zone)] = v
    
    zone_transitions_count = zone_transitions
    
    print("Restored tracking state from checkpoint")

def resize_frame(frame, max_width=1920, max_height=1080):
    """Resize frame to fit screen while maintaining aspect ratio"""
    height, width = frame.shape[:2]

    # Calculate scaling factor
    width_scale = max_width / width
    height_scale = max_height / height
    scale = min(width_scale, height_scale)

    # Only resize if image is larger than max dimensions
    if width > max_width or height > max_height:
        new_width = int(width * scale)
        new_height = int(height * scale)
        # Use INTER_AREA for downscaling (better quality)
        return cv2.resize(frame, (new_width, new_height),
                          interpolation=cv2.INTER_AREA)
    return frame


# Single video mode - set to empty string to disable single video mode
VIDEO_PATH = ""

# Batch processing mode
VIDEOS_DIRECTORY = "videos/"  # Directory containing all your videos
SUPPORTED_EXTENSIONS = ['.mp4', '.avi', '.mov', '.mkv', '.webm']

MODEL_PATH = "yolov8x.pt" 
TRACKER = "botsort.yaml"
ZONES_FILE = "zones.json"

# Detection settings
CONF_THRESHOLD = 0.15  # Lower threshold to catch more potential vehicles
IOU_THRESHOLD = 0.35   # Better for distinguishing vehicles in crowded scenes
IMG_SIZE = 1280        # Balanced for quality and performance

# UI settings
FRAME_SKIP = 10  # Process more frames for better tracking
SHOW_LABELS = True
SHOW_ZONES = True
SHOW_TRACKS = True
UI_TABLE_WIDTH = 300
MAX_DISPLAY_WIDTH = 1920
MAX_DISPLAY_HEIGHT = 1080

# GPU settings
USE_GPU = True
DEVICE = 'cuda:0' if USE_GPU and torch.cuda.is_available() else 'cpu'
HALF_PRECISION = True

# Tracking persistence settings
TRACK_BUFFER = 90  # Frames to keep a track alive through occlusions
MAX_TRACK_AGE = 45  # Maximum age before removing a track

# Enhanced settings
USE_ENHANCED_PREPROCESSING = True  # Enable advanced preprocessing
USE_MULTI_SCALE_DETECTION = False  # Disable multi-scale detection
USE_FRAME_ACCUMULATION = False     # Disable frame accumulation
ACCUMULATION_FRAMES = 3            # Number of frames to accumulate

# Better detection for crowded scenes
TRACK_CLEANUP_INTERVAL = 300  # Frames between trail cleanup operations
MAX_TRAIL_LENGTH = 30  # Maximum length of trail history to prevent memory issues
SAVE_CHECKPOINT_INTERVAL = 60  # Save checkpoint every 60 seconds

# Tracker parameters to pass directly
TRACKER_PARAMS = {
    "track_high_thresh": 0.2,
    "track_low_thresh": 0.1,
    "new_track_thresh": 0.2,
    "track_buffer": 60,
    "match_thresh": 0.8
}

CHECKPOINT_FILE = "checkpoint.json"  # File to store progress

#############################
# 2. ZONE EDITOR
#############################


class ZoneEditor:
    def __init__(self, video_path):
        self.video_path = video_path
        self.zones = {}
        self.main_zone = None
        self.current_zone_name = ""
        self.current_zone_points = []
        self.drawing = False
        self.editing_main_zone = False
        self.frame = None
        self.original_frame = None  # Store original frame
        self.preview_frame = None
        self.deleted_zones = {}  # Store deleted zones for potential recovery
        self.auto_save = True    # Auto-save zones when continuing
        self.width_scale = 1.0   # Scale factor for width
        self.height_scale = 1.0  # Scale factor for height

        # Crop area parameters
        self.crop_mode = False
        self.crop_start = None
        self.crop_end = None
        self.crop_dragging = False
        self.has_crop = False

        # UI related variables
        self.show_zone_table = True
        self.zone_table_width = 300
        self.zone_table_height = 600
        self.table_scroll_offset = 0
        self.selected_zone_index = -1  # -1 means no selection
        self.input_mode = False
        self.input_text = ""
        self.show_buttons = True
        self.zone_table_frame = None  # Frame for zone menu window

        # Auto-load zones on start if available
        if os.path.exists(ZONES_FILE):
            print(
                f"Found existing zones file ({ZONES_FILE}). Loading automatically.")
            self.load_zones()

    def mouse_callback(self, event, x, y, flags, param):
        # Convert mouse coordinates back to original frame coordinates
        orig_x = int(x / self.width_scale)
        orig_y = int(y / self.height_scale)

        # Handle crop area selection
        if self.crop_mode:
            if event == cv2.EVENT_LBUTTONDOWN:
                self.crop_start = (orig_x, orig_y)
                self.crop_end = (orig_x, orig_y)
                self.crop_dragging = True
            elif event == cv2.EVENT_MOUSEMOVE and self.crop_dragging:
                self.crop_end = (orig_x, orig_y)
            elif event == cv2.EVENT_LBUTTONUP:
                self.crop_dragging = False
                if self.crop_start and self.crop_end:
                    # Ensure start is top-left, end is bottom-right
                    x1 = min(self.crop_start[0], self.crop_end[0])
                    y1 = min(self.crop_start[1], self.crop_end[1])
                    x2 = max(self.crop_start[0], self.crop_end[0])
                    y2 = max(self.crop_start[1], self.crop_end[1])

                    # Set normalized crop coordinates (for different
                    # resolutions)
                    self.crop_start = (x1, y1)
                    self.crop_end = (x2, y2)
                    self.has_crop = True
                    print(f"Crop area set: ({x1}, {y1}) to ({x2}, {y2})")
                    if self.auto_save:
                        self.save_zones()
            return

        # Normal zone drawing
        if event == cv2.EVENT_LBUTTONDOWN:
            if not self.input_mode:  # Don't add points when in text input mode
                self.current_zone_points.append((orig_x, orig_y))
                self.drawing = True
        elif event == cv2.EVENT_RBUTTONDOWN and self.drawing and len(self.current_zone_points) > 2:
            # Complete the polygon with right-click
            if self.editing_main_zone:
                self.main_zone = Polygon(self.current_zone_points)
                print(
                    f"Main zone created with {len(self.current_zone_points)} points")
                self.editing_main_zone = False
                if self.auto_save:
                    self.save_zones()
            else:
                self.zones[self.current_zone_name] = Polygon(
                    self.current_zone_points)
                print(
                    f"Zone '{self.current_zone_name}' created with {len(self.current_zone_points)} points")
                if self.auto_save:
                    self.save_zones()

            self.current_zone_points = []
            self.drawing = False

    def zone_table_mouse_callback(self, event, x, y, flags, param):
        """Handle clicks in the zone table window"""
        # Check button clicks at the top of the table
        button_height = 40
        button_width = self.zone_table_width / 4

        if y < button_height:
            # Add Main Zone button
            if x < button_width:
                if event == cv2.EVENT_LBUTTONDOWN:
                    self.start_main_zone()
            # Add Zone button
            elif x < button_width * 2:
                if event == cv2.EVENT_LBUTTONDOWN:
                    self.start_add_zone()
            # Delete button
            elif x < button_width * 3:
                if event == cv2.EVENT_LBUTTONDOWN:
                    self.delete_selected_zone()
            # Continue button
            elif x <= button_width * 4:
                if event == cv2.EVENT_LBUTTONDOWN:
                    self.continue_to_detection()
            return

        # Zone list area
        list_start_y = button_height + 30  # Header height
        row_height = 30

        # Calculate which zone was clicked (accounting for scroll)
        if y >= list_start_y:
            row_index = (y - list_start_y) // row_height + \
                self.table_scroll_offset

            # Check if Main Zone was clicked
            if self.main_zone is not None and row_index == 0:
                if event == cv2.EVENT_LBUTTONDOWN:
                    self.selected_zone_index = 0
            # Check if a regular zone was clicked
            elif self.main_zone is not None and 0 < row_index <= len(self.zones):
                if event == cv2.EVENT_LBUTTONDOWN:
                    self.selected_zone_index = row_index
            # If no main zone, adjust index
            elif self.main_zone is None and row_index < len(self.zones):
                if event == cv2.EVENT_LBUTTONDOWN:
                    self.selected_zone_index = row_index + 1  # +1 to account for missing main zone

    def start_main_zone(self):
        """Start drawing the main zone"""
        if self.main_zone is not None:
            overwrite = input("Main zone already exists. Overwrite? (y/n): ")
            if overwrite.lower() != 'y':
                return

        self.current_zone_points = []
        self.editing_main_zone = True
        self.drawing = True
        print("Drawing main zone. Left-click to add points, right-click to complete.")

    def start_add_zone(self):
        """Start adding a new zone with visual name input"""
        if self.main_zone is None:
            print("Please define the main zone first")
            return

        self.input_mode = True
        self.input_text = ""
        print("Enter zone name. Press ENTER when done.")

    def complete_zone_name(self):
        """Complete the zone naming process"""
        if self.input_text.strip() == "":
            print("Zone name cannot be empty")
            self.input_mode = False
            return

        if self.input_text in self.zones:
            print("Zone name already exists")
            self.input_mode = False
            return

        self.current_zone_name = self.input_text
        self.input_mode = False
        self.input_text = ""
        self.current_zone_points = []
        self.drawing = True
        print(
            f"Drawing zone '{self.current_zone_name}'. Left-click to add points, right-click to complete.")

    def delete_selected_zone(self):
        """Delete the currently selected zone"""
        if self.selected_zone_index == -1:
            print("No zone selected")
            return

        if self.selected_zone_index == 0 and self.main_zone is not None:
            # Delete main zone
            self.deleted_zones["main_zone"] = self.main_zone
            self.main_zone = None
            print("Main zone deleted")
            self.selected_zone_index = -1
            if self.auto_save:
                self.save_zones()
        elif self.main_zone is not None and 0 < self.selected_zone_index <= len(self.zones):
            # Delete regular zone
            zone_name = list(self.zones.keys())[self.selected_zone_index - 1]
            self.deleted_zones[zone_name] = self.zones[zone_name]
            del self.zones[zone_name]
            print(f"Zone '{zone_name}' deleted")
            self.selected_zone_index = -1
            if self.auto_save:
                self.save_zones()
        elif self.main_zone is None and self.selected_zone_index < len(self.zones):
            # No main zone scenario
            zone_name = list(self.zones.keys())[self.selected_zone_index]
            self.deleted_zones[zone_name] = self.zones[zone_name]
            del self.zones[zone_name]
            print(f"Zone '{zone_name}' deleted")
            self.selected_zone_index = -1
            if self.auto_save:
                self.save_zones()

    def continue_to_detection(self):
        """Continue to detection phase"""
        if self.main_zone is not None:
            if not self.zones:
                add_anyway = input(
                    "No zones defined (besides main). Continue anyway? (y/n): ")
                if add_anyway.lower() != 'y':
                    return
            if self.auto_save:
                self.save_zones()
            self.should_continue = True  # Set flag to exit the loop
        else:
            print("Please define at least the main zone before continuing.")

    def draw_zones(self, frame):
        # Draw the main zone
        if self.main_zone is not None:
            coords = list(self.main_zone.exterior.coords)
            for i in range(len(coords) - 1):
                pt1 = (int(coords[i][0] * self.width_scale),
                       int(coords[i][1] * self.height_scale))
                pt2 = (int(coords[i + 1][0] * self.width_scale),
                       int(coords[i + 1][1] * self.height_scale))
                cv2.line(frame, pt1, pt2, (0, 0, 255), 2)  # Main zone in red
            cv2.putText(frame,
                        "MAIN ZONE",
                        (int(coords[0][0] * self.width_scale),
                         int(coords[0][1] * self.height_scale) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0,
                            0,
                            255),
                        2)

        # Draw other zones
        for zone_name, zone_poly in self.zones.items():
            coords = list(zone_poly.exterior.coords)
            for i in range(len(coords) - 1):
                pt1 = (int(coords[i][0] * self.width_scale),
                       int(coords[i][1] * self.height_scale))
                pt2 = (int(coords[i + 1][0] * self.width_scale),
                       int(coords[i + 1][1] * self.height_scale))
                # Regular zones in blue
                cv2.line(frame, pt1, pt2, (255, 0, 0), 2)
            cv2.putText(frame,
                        zone_name,
                        (int(coords[0][0] * self.width_scale),
                         int(coords[0][1] * self.height_scale) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255,
                            0,
                            0),
                        2)

        # Draw the current polygon being created
        if self.drawing and len(self.current_zone_points) > 0:
            for i in range(len(self.current_zone_points) - 1):
                cv2.line(frame,
                         (int(self.current_zone_points[i][0] * self.width_scale),
                          int(self.current_zone_points[i][1] * self.height_scale)),
                         (int(self.current_zone_points[i + 1][0] * self.width_scale),
                             int(self.current_zone_points[i + 1][1] * self.height_scale)),
                         (0,
                             255,
                             0),
                         2)

            # Line from last point to first if we have more than 2 points
            if len(self.current_zone_points) > 2:
                cv2.line(frame,
                         (int(self.current_zone_points[-1][0] * self.width_scale),
                          int(self.current_zone_points[-1][1] * self.height_scale)),
                         (int(self.current_zone_points[0][0] * self.width_scale),
                             int(self.current_zone_points[0][1] * self.height_scale)),
                         (0,
                             255,
                             0),
                         2)

            # Draw dots for each point to make them more visible
            for point in self.current_zone_points:
                cv2.circle(frame, (int(point[0] *
                                       self.width_scale), int(point[1] *
                           self.height_scale)), 3, (0, 255, 255), -
                           1)

        # Draw crop area
        if self.crop_mode or self.has_crop:
            if self.crop_start and self.crop_end:
                x1 = int(self.crop_start[0] * self.width_scale)
                y1 = int(self.crop_start[1] * self.height_scale)
                x2 = int(self.crop_end[0] * self.width_scale)
                y2 = int(self.crop_end[1] * self.height_scale)

                # Draw rectangle
                cv2.rectangle(frame, (x1, y1), (x2, y2),
                              (255, 255, 0), 2)  # Yellow rectangle

                # Draw diagonal crosshair
                cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 0), 1)
                cv2.line(frame, (x1, y2), (x2, y1), (255, 255, 0), 1)

                # Label
                cv2.putText(frame, "CROP AREA", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    def create_zone_table_window(self):
        """Create a separate window for the zone table"""
        self.zone_table_frame = np.zeros((self.zone_table_height, self.zone_table_width, 3), dtype=np.uint8)
        self.zone_table_frame.fill(50)  # Dark gray background
        cv2.namedWindow('Zone Table')
        cv2.moveWindow('Zone Table', 50 + self.frame.shape[1] + 10, 50)  # Position next to main window
        cv2.setMouseCallback('Zone Table', self.zone_table_mouse_callback)

    def draw_zone_table(self):
        """Draw the zone table in a separate window"""
        if not self.show_zone_table or self.zone_table_frame is None:
            return

        # Clear the table background
        self.zone_table_frame.fill(50)  # Dark gray background

        # Draw button row
        button_height = 40
        button_width = self.zone_table_width / 4

        # Main Zone button
        cv2.rectangle(
            self.zone_table_frame, (0, 0), (int(
                button_width), button_height), (100, 100, 100), -1)
        cv2.putText(self.zone_table_frame, "Main Zone", (5, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Add Zone button
        cv2.rectangle(self.zone_table_frame, (int(
                button_width), 0), (int(button_width *
                                                          2), button_height), (100, 100, 100), -
                      1)
        cv2.putText(self.zone_table_frame, "Add Zone", (int(button_width) + 5,
                    25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Delete button
        cv2.rectangle(self.zone_table_frame, (int(
                                  button_width *
                2), 0), (int(button_width *
                                               3), button_height), (100, 100, 100), -
                      1)
        cv2.putText(
            self.zone_table_frame,
            "Delete",
            (int(
                button_width *
                2) +
                5,
                25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255,
             255,
             255),
            1)

        # Continue button
        cv2.rectangle(self.zone_table_frame, (int(button_width * 3), 0),
                      (self.zone_table_width, button_height), (0, 100, 0), -1)
        cv2.putText(
            self.zone_table_frame,
            "Continue",
            (int(
                button_width *
                3) +
                5,
                25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255,
             255,
             255),
            1)

        # Table header
        header_y = button_height
        cv2.rectangle(self.zone_table_frame, (0, header_y),
                      (self.zone_table_width, header_y + 30), (70, 70, 70), -1)

        # Draw column dividers and headers
        name_col_width = int(self.zone_table_width * 0.7)
        points_col_width = self.zone_table_width - name_col_width

        # Column headers with better spacing
        cv2.putText(self.zone_table_frame, "Zone Name", (10, header_y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Draw vertical divider
        divider_x = name_col_width
        cv2.line(self.zone_table_frame, (divider_x, header_y),
                 (divider_x, self.zone_table_height), (100, 100, 100), 1)
        cv2.putText(self.zone_table_frame, "Points", (divider_x + 10, header_y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Zone list
        list_start_y = header_y + 30
        row_height = 30
        visible_rows = (self.zone_table_height - list_start_y) // row_height

        # Prepare list of all zones (main zone first, then others)
        all_zones = []
        if self.main_zone is not None:
            all_zones.append(("MAIN ZONE", self.main_zone))
        for name, poly in self.zones.items():
            all_zones.append((name, poly))

        # Draw rows with scroll offset
        for i in range(min(visible_rows, len(
                all_zones) - self.table_scroll_offset)):
            zone_idx = i + self.table_scroll_offset
            if zone_idx >= len(all_zones):
                break

            row_y = list_start_y + i * row_height
            name, poly = all_zones[zone_idx]

            # Horizontal divider between rows
            cv2.line(self.zone_table_frame, (0, row_y),
                     (self.zone_table_width, row_y), (100, 100, 100), 1)

            # Highlight selected row
            if (self.main_zone is not None and zone_idx == self.selected_zone_index) or (
                    self.main_zone is None and zone_idx + 1 == self.selected_zone_index):
                cv2.rectangle(
                    self.zone_table_frame, (0, row_y), (self.zone_table_width, row_y + row_height), (100, 100, 160), -1)
            else:
                cv2.rectangle(
                    self.zone_table_frame, (0, row_y), (self.zone_table_width, row_y + row_height), (60, 60, 60), -1)

            # Zone name
            cv2.putText(self.zone_table_frame, name, (10, row_y + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            # Vertical divider
            cv2.line(self.zone_table_frame, (divider_x, row_y), (divider_x,
                     row_y + row_height), (100, 100, 100), 1)

            # Point count
            # -1 because first/last points are the same
            point_count = len(list(poly.exterior.coords)) - 1
            cv2.putText(self.zone_table_frame, str(point_count), (divider_x + 10, row_y + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Draw scroll indicators if needed
        if self.table_scroll_offset > 0:
            cv2.putText(
                self.zone_table_frame,
                "▲",
                (self.zone_table_width -
                 30,
                 list_start_y +
                 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255,
                 255,
                 255),
                1)

        if self.table_scroll_offset + visible_rows < len(all_zones):
            cv2.putText(
                self.zone_table_frame,
                "▼",
                (self.zone_table_width -
                 30,
                 self.zone_table_height -
                 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255,
                 255,
                 255),
                1)

        # Text input box for zone naming
        if self.input_mode:
            input_box_y = int(self.zone_table_height / 2)
            input_box_width = 300
            input_box_x = int(
                (self.zone_table_width - input_box_width) / 2)

            # Draw input box
            cv2.rectangle(self.zone_table_frame, (input_box_x, input_box_y -
                                  40), (input_box_x +
                                        input_box_width, input_box_y +
                                        40), (30, 30, 30), -
                          1)
            cv2.rectangle(self.zone_table_frame, (input_box_x, input_box_y -
                                  40), (input_box_x +
                                        input_box_width, input_box_y +
                                        40), (100, 100, 100), 2)

            # Title
            cv2.putText(
                self.zone_table_frame,
                "Enter Zone Name:",
                (input_box_x + 10,
                 input_box_y - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255,
                 255,
                 255),
                1)

            # Input text
            cv2.putText(
                self.zone_table_frame,
                self.input_text + "|",
                (input_box_x + 10,
                 input_box_y + 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255,
                 255,
                 255),
                1)

            # Instructions
            cv2.putText(self.zone_table_frame, "Press ENTER when done, ESC to cancel",
                        (input_box_x + 10, input_box_y + 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        # Display the zone table window
        cv2.imshow('Zone Table', self.zone_table_frame)

    def undo_last_point(self):
        """Remove the last added point when drawing a zone"""
        if self.drawing and len(self.current_zone_points) > 0:
            removed_point = self.current_zone_points.pop()
            print(f"Removed point {removed_point}")
            return True
        return False

    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"Error: Cannot open video {self.video_path}")
            return False

        # Get first frame for drawing
        ret, self.original_frame = cap.read()
        if not ret:
            print("Error: Cannot read frame from video")
            return False

        # Store original dimensions
        original_height, original_width = self.original_frame.shape[:2]

        # Resize frame to fit screen
        self.frame = resize_frame(
            self.original_frame.copy(),
            MAX_DISPLAY_WIDTH,
            MAX_DISPLAY_HEIGHT)

        # Calculate scaling factors
        new_height, new_width = self.frame.shape[:2]
        self.width_scale = new_width / original_width
        self.height_scale = new_height / original_height

        # Create main window for video/zones
        cv2.namedWindow('Zone Editor')
        cv2.setMouseCallback('Zone Editor', self.mouse_callback)
        
        # Create separate window for zone table
        self.create_zone_table_window()

        self.should_continue = False

        while True:
            self.preview_frame = self.frame.copy()

            # Draw zones on main window
            self.draw_zones(self.preview_frame)
            
            # Draw zone table in separate window
            self.draw_zone_table()

            # Show current mode
            mode_text = "EDITING: "
            if self.crop_mode:
                mode_text += "CROP AREA"
            elif self.editing_main_zone:
                mode_text += "MAIN ZONE"
            elif self.drawing:
                mode_text += f"ZONE '{self.current_zone_name}'"
            elif self.input_mode:
                mode_text += "ZONE NAME"
            else:
                mode_text += "NONE"
            cv2.putText(
                self.preview_frame,
                mode_text,
                (10,
                 self.frame.shape[0] -
                 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0,
                 255,
                 255),
                2)

            # Display auto-save status
            auto_save_text = "AUTO-SAVE: ON" if self.auto_save else "AUTO-SAVE: OFF"
            cv2.putText(
                self.preview_frame,
                auto_save_text,
                (10,
                 self.frame.shape[0] -
                 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0,
                 255,
                 255),
                2)

            # Display keyboard shortcuts (when not in input mode)
            if not self.input_mode:
                cv2.putText(
                    self.preview_frame,
                    "U: Undo Point | T: Toggle Auto-save | C: Crop Mode | R: Reset Crop | ESC: Exit",
                    (10,
                     30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255,
                     255,
                     255),
                    2)

            cv2.imshow('Zone Editor', self.preview_frame)

            key = cv2.waitKey(1) & 0xFF

            # Handle text input mode
            if self.input_mode:
                if key == 13:  # Enter key
                    self.complete_zone_name()
                elif key == 27:  # Escape key
                    self.input_mode = False
                    self.input_text = ""
                elif key == 8:  # Backspace
                    self.input_text = self.input_text[:- \
                        1] if self.input_text else ""
                elif 32 <= key <= 126:  # Printable ASCII
                    self.input_text += chr(key)
                continue

            # Normal mode key handling
            if key == 27:  # ESC
                break
            elif key == ord('u'):  # Undo last point
                self.undo_last_point()
            elif key == ord('t'):  # Toggle auto-save
                self.auto_save = not self.auto_save
                print(
                    f"Auto-save {'enabled' if self.auto_save else 'disabled'}")
            elif key == ord('c'):  # Toggle crop mode
                self.crop_mode = not self.crop_mode
                if self.crop_mode:
                    print("Crop mode enabled. Click and drag to define crop area.")
                else:
                    print("Crop mode disabled.")
            elif key == ord('r'):  # Reset crop
                self.crop_start = None
                self.crop_end = None
                self.has_crop = False
                print("Crop area reset.")
                if self.auto_save:
                    self.save_zones()
            elif key == ord('m'):  # Main zone
                self.start_main_zone()
            elif key == ord('a'):  # Add zone
                self.start_add_zone()
            elif key == ord('d'):  # Delete selected zone
                self.delete_selected_zone()
            elif key == ord('s'):  # Save zones
                if self.main_zone is not None or self.zones:
                    self.save_zones()
                else:
                    print("Nothing to save. Please define zones first.")
            # Scroll controls
            elif key == ord('w'):  # Scroll up
                self.table_scroll_offset = max(0, self.table_scroll_offset - 1)
            elif key == ord('z'):  # Scroll down
                max_scroll = max(0, len(self.zones) +
                                 (1 if self.main_zone is not None else 0) - 5)
                self.table_scroll_offset = min(
                    max_scroll, self.table_scroll_offset + 1)

            # Check if we should continue to detection
            if self.should_continue:
                cv2.destroyAllWindows()
                return True

        cap.release()
        cv2.destroyAllWindows()
        return False

    def save_zones(self):
        # Convert polygons to lists of points for JSON serialization
        zones_data = {}
        if self.main_zone is not None:
            zones_data["main_zone"] = list(self.main_zone.exterior.coords)

        for name, polygon in self.zones.items():
            zones_data[name] = list(polygon.exterior.coords)

        # Add crop area if defined
        if self.has_crop and self.crop_start and self.crop_end:
            zones_data["crop_area"] = {
                "x1": self.crop_start[0],
                "y1": self.crop_start[1],
                "x2": self.crop_end[0],
                "y2": self.crop_end[1]
            }

        with open(ZONES_FILE, 'w') as f:
            json.dump(zones_data, f)

        print(f"Zones saved to {ZONES_FILE}")

    def load_zones(self):
        try:
            with open(ZONES_FILE, 'r') as f:
                zones_data = json.load(f)

            # Convert points back to Polygons
            if "main_zone" in zones_data:
                self.main_zone = Polygon(zones_data["main_zone"])
                zones_data.pop("main_zone")

            # Load crop area if available
            if "crop_area" in zones_data:
                crop_data = zones_data.pop("crop_area")
                self.crop_start = (crop_data["x1"], crop_data["y1"])
                self.crop_end = (crop_data["x2"], crop_data["y2"])
                self.has_crop = True
                print(
                    f"Loaded crop area: ({self.crop_start}) to ({self.crop_end})")

            for name, points in zones_data.items():
                self.zones[name] = Polygon(points)

            print(
                f"Loaded {len(self.zones) + (1 if self.main_zone is not None else 0)} zones")
            return True
        except Exception as e:
            print(f"Error loading zones: {e}")
            return False

#############################
# 3. PREPROCESS FUNCTION
#############################


def apply_clahe(frame):
    """Enhances contrast in dark videos"""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)

    merged = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    return enhanced

def enhanced_preprocess(frame):
    """Enhanced preprocessing pipeline with GPU optimization when possible"""
    if not USE_ENHANCED_PREPROCESSING:
        return frame
    
    try:
        # Option 1: For CUDA-enabled OpenCV builds only
        if hasattr(cv2, 'cuda') and cv2.cuda.getCudaEnabledDeviceCount() > 0:
            # Move processing to GPU where possible
            gpu_frame = cv2.cuda_GpuMat()
            gpu_frame.upload(frame)
            
            # Use GPU-accelerated versions of filters when available
            # Blur for noise reduction (faster than CPU denoising)
            gpu_frame = cv2.cuda.blur(gpu_frame, (5, 5))
            
            # Download for operations that must be done on CPU
            cpu_frame = gpu_frame.download()
            
            # Step 2: Enhanced CLAHE with better parameters (CPU operation)
            lab = cv2.cvtColor(cpu_frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            merged = cv2.merge((cl, a, b))
            enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
            
            return enhanced
    except Exception as e:
        # If GPU acceleration fails, fallback to optimized CPU version
        print(f"GPU preprocessing not available: {e}")
    
    # CPU optimization: use a faster but still effective pipeline
    # 1. Use a faster bilateral filter instead of fastNlMeansDenoisingColored
    denoised = cv2.bilateralFilter(frame, 9, 75, 75)  # Much faster than fastNlMeans
    
    # 2. Enhanced CLAHE with better parameters
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    
    # Skip sharpening as it's CPU-intensive and provides marginal benefits
    # Just do brightness/contrast adjustment which is fast
    alpha = 1.2  # Contrast control
    beta = 10    # Brightness control
    adjusted = cv2.convertScaleAbs(enhanced, alpha=alpha, beta=beta)
    
    return adjusted

class FrameAccumulator:
    """Accumulates frames to reduce noise in static camera footage"""
    def __init__(self, max_frames=3):
        self.max_frames = max_frames
        self.frames = []
        
    def add_frame(self, frame):
        """Add a frame to the accumulator"""
        if frame is None:
            return
            
        # Make sure frame is 8-bit for consistent accumulation
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
            
        # Store a deep copy to prevent modification
        if len(self.frames) >= self.max_frames:
            self.frames.pop(0)
        self.frames.append(frame.copy())
        
    def get_accumulated(self):
        """Get accumulated frame (average of stored frames)"""
        if not self.frames:
            return None
            
        if len(self.frames) == 1:
            return self.frames[0]
        
        try:    
            # Create an accumulated frame by averaging
            acc = np.zeros_like(self.frames[0], dtype=np.float32)
            for frame in self.frames:
                acc += frame.astype(np.float32)
            acc /= len(self.frames)
            
            # Convert back to uint8 for OpenCV compatibility
            return np.clip(acc, 0, 255).astype(np.uint8)
        except Exception as e:
            print(f"Error in frame accumulation: {e}")
            # If anything goes wrong, return the most recent frame
            return self.frames[-1] if self.frames else None

def multi_scale_detect(model, frame, conf_threshold, iou_threshold):
    """Run detection at multiple scales with GPU optimization"""
    # Base scale detection (original frame)
    results_original = model.predict(
        source=frame, 
        conf=conf_threshold, 
        iou=iou_threshold, 
        device=DEVICE,
        verbose=False
    )
    
    # Only perform multi-scale if enabled
    if not USE_MULTI_SCALE_DETECTION:
        return results_original
    
    # Check if we need additional scales - if we already have detections, no need for multi-scale
    if len(results_original[0].boxes) > 3:
        return results_original
        
    try:
        # Only do upscaled detection for small vehicles
        h, w = frame.shape[:2]
        resized_up = cv2.resize(frame, (int(w*1.5), int(h*1.5)))
        
        # Use more permissive confidence threshold for scaled detection
        results_up = model.predict(
            source=resized_up, 
            conf=conf_threshold-0.05, 
            iou=iou_threshold,
            device=DEVICE,
            verbose=False
        )
        
        # Return whichever found more objects, but don't try to merge them
        if len(results_up[0].boxes) > len(results_original[0].boxes):
            return results_up
        return results_original
    except Exception as e:
        print(f"Error in multi-scale detection: {e}")
        # If anything goes wrong, return original results
        return results_original

#############################
# 4. DATA STRUCTURES
#############################
vehicle_paths = {}       # Path history
last_known_zone = {}     # Current zone
zone_transitions_count = {}  # Traffic flow

# Add data structures for improved tracking
vehicle_last_seen = {}    # Frame number when vehicle was last seen
vehicle_positions = {}    # Last known positions of vehicles
vehicle_predictions = {}  # Predicted positions of occluded vehicles
PREDICTION_MAX_FRAMES = 10  # Max frames to predict position
is_first_frame = True     # Flag for first frame processing

# Structures for vehicle trails
vehicle_trail_history = {}  # Store position history for each vehicle ID
vehicle_arrived = {}        # Store time when vehicle reached destination
vehicle_completed = set()   # Vehicles that have completed their journey
# All possible destination zones
destination_zones = {"Gate", "PL1", "PL2", "PL3", "SQL"}
TRAIL_FADE_SECONDS = 2.0    # Time in seconds for trail to fade after arrival
TRAIL_REMOVE_SECONDS = 3.0  # Time in seconds before removing trail completely
MIN_ZONE_STAY_FRAMES = 30   # Minimum frames in destination zone to consider "arrived"

# Add these new global variables after the existing global declarations
# Time interval tracking
time_interval_minutes = 10  # Summary interval in minutes
time_segment_transitions = {}  # Transitions within current time segment
current_time_segment = 0  # Current time segment (0-10 min, 11-20 min, etc.)
last_segment_time = 0  # Last recorded segment time

def save_time_segment_summary(video_name, start_minute, end_minute, transitions, results_dir="results"):
    """Save the time segment summary to a cumulative file"""
    # Create the summary file if it doesn't exist
    summary_file = os.path.join(results_dir, "time_segment_summaries.txt")
    
    # Format the transitions data
    transition_lines = []
    for (from_zone, to_zone), count in transitions.items():
        transition_lines.append(f"{from_zone} -> {to_zone}: {count}")
    
    # Sort by count (descending)
    transition_lines.sort(key=lambda x: int(x.split(": ")[1]), reverse=True)
    
    # Only write if there are transitions to report
    if transition_lines:
        with open(summary_file, "a") as f:
            f.write(f"\nVideo name: {video_name}\n")
            f.write(f"{start_minute}-{end_minute}\n")
            for line in transition_lines:
                f.write(f"{line}\n")
            f.write("\n")  # Extra line for separation
        
        print(f"Saved time segment summary for {start_minute}-{end_minute} minutes")
    else:
        print(f"No transitions in time segment {start_minute}-{end_minute} minutes")

#############################
# 5. LOAD MODEL
#############################
model = YOLO(MODEL_PATH)
model.conf = CONF_THRESHOLD
model.iou = IOU_THRESHOLD

#############################
# 6. PARAMETER UI FUNCTIONS
#############################


def create_parameter_window(width=400, height=600, tracking_params=None):
    """Create a window to display and control parameters"""
    params_img = np.zeros((height, width, 3), dtype=np.uint8)
    params_img.fill(50)  # Dark gray background

    # Create window
    cv2.namedWindow("Parameters", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Parameters", width, height)

    # Position the parameters window
    # Position between main and summary windows
    cv2.moveWindow("Parameters", 550, 50)

    return params_img


def create_summary_window(width=400, height=500):
    """Create a window to display tracking summary"""
    summary_img = np.zeros((height, width, 3), dtype=np.uint8)
    summary_img.fill(50)  # Dark gray background

    # Create window
    cv2.namedWindow("Tracking Summary", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Tracking Summary", width, height)

    return summary_img


def update_summary_window(summary_img, zone_transitions, vehicle_paths, zones):
    """Update the summary window with tracking data"""
    # Clear the window (keep background color)
    summary_img.fill(50)

    # Draw title
    cv2.rectangle(summary_img, (0, 0),
                  (summary_img.shape[1], 40), (70, 70, 70), -1)
    cv2.putText(summary_img, "TRACKING SUMMARY", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Display zone transition counts
    y_pos = 60
    cv2.putText(summary_img, "Zone Transitions:", (10, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    y_pos += 30
    for (from_zone, to_zone), count in zone_transitions.items():
        text = f"{from_zone} -> {to_zone}: {count}"
        cv2.putText(summary_img, text, (20, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y_pos += 20

        # Prevent overflow
        if y_pos > summary_img.shape[0] - 50:
            cv2.putText(summary_img, "...", (20, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            break

    cv2.imshow("Tracking Summary", summary_img)
    return summary_img


def handle_param_mouse(event, x, y, flags, param, tracking_params):
    """Handle mouse events for parameter window"""
    # Reset active slider on button release or right-click
    if event == cv2.EVENT_LBUTTONUP or event == cv2.EVENT_RBUTTONDOWN:
        tracking_params["_active_slider"] = None
        return

    # Only handle left-click events and mouse move with active slider
    if event != cv2.EVENT_LBUTTONDOWN and (
            event != cv2.EVENT_MOUSEMOVE or tracking_params["_active_slider"] is None):
        return

    # Check for active slider drag
    if event == cv2.EVENT_MOUSEMOVE and tracking_params["_active_slider"] is not None:
        slider_name = tracking_params["_active_slider"]
        slider_coords = tracking_params[slider_name]["slider_coords"]
        x_start, _, x_end, _ = slider_coords

        # Calculate new value based on position
        param_info = tracking_params[slider_name]
        value_range = param_info["max"] - param_info["min"]
        slider_width = x_end - x_start
        rel_pos = max(0, min(slider_width, x - x_start))
        new_value = param_info["min"] + (rel_pos / slider_width) * value_range

        # Discretize to steps if needed
        if "step" in param_info and param_info["step"] > 0:
            new_value = round(
                new_value / param_info["step"]) * param_info["step"]

        # Update the value
        tracking_params[slider_name]["value"] = new_value
        return

    # Handle button clicks
    for param_name, param_info in tracking_params.items():
        # Skip non-param entries
        if param_name.startswith('_'):
            continue

        # Check button clicks for boolean parameters
        if "button_coords" in param_info:
            x1, y1, x2, y2 = param_info["button_coords"]
            if x1 <= x <= x2 and y1 <= y <= y2 and event == cv2.EVENT_LBUTTONDOWN:
                # Toggle the value
                param_info["value"] = 1 - param_info["value"]
                return

        # Check slider clicks for numeric parameters
        if "slider_coords" in param_info:
            x1, y1, x2, y2 = param_info["slider_coords"]
            if x1 <= x <= x2 and y1 <= y <= y2:
                if event == cv2.EVENT_LBUTTONDOWN:
                    # Set as active slider and update value
                    tracking_params["_active_slider"] = param_name

                    # Calculate new value based on position
                    value_range = param_info["max"] - param_info["min"]
                    slider_width = x2 - x1
                    rel_pos = max(0, min(slider_width, x - x1))
                    new_value = param_info["min"] + \
                        (rel_pos / slider_width) * value_range

                    # Discretize to steps if needed
                    if "step" in param_info and param_info["step"] > 0:
                        new_value = round(
                            new_value / param_info["step"]) * param_info["step"]

                    # Update the value
                    param_info["value"] = new_value
                    return


def draw_param_ui(frame, params, separate_window=False):
    """Draw parameter control UI, either on main frame or separate window"""
    if separate_window:
        # Draw on a separate parameter window
        params_img = np.zeros((600, 400, 3), dtype=np.uint8)
        params_img.fill(50)  # Dark gray background

        # Draw title and GPU info
        cv2.rectangle(params_img, (0, 0), (400, 40), (70, 70, 70), -1)
        cv2.putText(params_img, "PARAMETERS", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Display GPU/CPU info if available
        if "_gpu_info" in params:
            gpu_text = f"Device: {params['_gpu_info']}"
            cv2.putText(params_img, gpu_text, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Draw parameters
        param_names = [
            name for name in params.keys() if not name.startswith('_')]
        row_height = 50
        slider_padding = 80

        for i, name in enumerate(param_names):
            row_y = 80 + i * row_height
            param = params[name]

            # Row background
            cv2.rectangle(params_img, (0, row_y),
                          (400, row_y + row_height), (60, 60, 60), -1)

            # Parameter name
            cv2.putText(params_img, name, (10, row_y + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            # Draw slider track
            slider_x_start = slider_padding
            slider_x_end = 350
            slider_width = slider_x_end - slider_x_start
            slider_y = row_y + 30

            # Boolean parameters (toggle buttons)
            if name in ["Show Labels", "Show Zones", "Show Tracks", "Show Trails", "Enhanced Preproc"]:
                # Draw toggle button
                button_width = 60
                button_x = slider_x_start + 40

                if param["value"] == 1:  # ON
                    button_color = (0, 200, 0)  # Green
                    button_text = "ON"
                else:  # OFF
                    button_color = (0, 0, 200)  # Red
                    button_text = "OFF"

                cv2.rectangle(params_img, (button_x, row_y + 15),
                              (button_x + button_width, row_y + 45), button_color, -1)
                cv2.rectangle(params_img, (button_x, row_y +
                                           15), (button_x +
                                                 button_width, row_y +
                                                 45), (100, 100, 100), 1)
                text_x = button_x + 15
                cv2.putText(params_img, button_text, (text_x, row_y + 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

                # Store button coordinates for mouse interaction
                param["button_coords"] = (
                    button_x, row_y + 15, button_x + button_width, row_y + 45)

            else:  # Numeric parameters with sliders
                # Draw slider track
                cv2.rectangle(params_img, (slider_x_start, slider_y - 5),
                              (slider_x_end, slider_y + 5), (100, 100, 100), -1)

                # Calculate slider position
                value_range = param["max"] - param["min"]
                if value_range == 0:  # Avoid division by zero
                    value_range = 1
                slider_pos = int(
                    slider_x_start + (param["value"] - param["min"]) / value_range * slider_width)

                # Draw slider handle
                handle_radius = 8
                cv2.circle(params_img, (slider_pos, slider_y),
                           handle_radius, (0, 200, 200), -1)
                cv2.circle(params_img, (slider_pos, slider_y),
                           handle_radius, (200, 200, 200), 1)

                # Store slider coordinates for mouse interaction
                param["slider_coords"] = (
                    slider_x_start, slider_y - 10, slider_x_end, slider_y + 10)
                param["slider_pos"] = slider_pos

                # Get value with correct type display
                if name in ["Frame Skip"]:
                    value_str = str(int(param["value"]))
                else:
                    value_str = f"{param['value']:.2f}"

                # Parameter value
                cv2.putText(
                    params_img,
                    value_str,
                    (slider_x_end + 10,
                     slider_y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255,
                     255,
                     255),
                    1)

        # Show the parameters window
        cv2.imshow("Parameters", params_img)
        return params_img
    else:
        # Original implementation for drawing on main frame
        table_width = UI_TABLE_WIDTH
        table_x = frame.shape[1] - table_width

        # Draw background
        cv2.rectangle(frame, (table_x, 0),
                      (frame.shape[1], frame.shape[0]), (50, 50, 50), -1)

        # Draw title and GPU info
        cv2.rectangle(frame, (table_x, 0),
                      (frame.shape[1], 40), (70, 70, 70), -1)
        cv2.putText(frame, "PARAMETERS", (table_x + 10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Display GPU/CPU info if available
        if "_gpu_info" in params:
            gpu_text = f"Device: {params['_gpu_info']}"
            cv2.putText(frame, gpu_text, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Draw parameters
        param_names = [
            name for name in params.keys() if not name.startswith('_')]
        row_height = 50
        slider_padding = 80

        for i, name in enumerate(param_names):
            row_y = 40 + i * row_height
            param = params[name]

            # Row background
            cv2.rectangle(frame, (table_x, row_y),
                          (frame.shape[1], row_y + row_height), (60, 60, 60), -1)

            # Parameter name
            cv2.putText(frame, name, (table_x + 10, row_y + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            # Draw slider track
            slider_x_start = table_x + slider_padding
            slider_x_end = table_x + table_width - 50
            slider_width = slider_x_end - slider_x_start
            slider_y = row_y + 30

            # Boolean parameters (toggle buttons)
            if name in ["Show Labels", "Show Zones", "Show Tracks", "Show Trails", "Enhanced Preproc"]:
                # Draw toggle button
                button_width = 60
                button_x = slider_x_start + 40

                if param["value"] == 1:  # ON
                    button_color = (0, 200, 0)  # Green
                    button_text = "ON"
                else:  # OFF
                    button_color = (0, 0, 200)  # Red
                    button_text = "OFF"

                cv2.rectangle(frame, (button_x, row_y + 15),
                              (button_x + button_width, row_y + 45), button_color, -1)
                cv2.rectangle(frame, (button_x, row_y +
                                      15), (button_x +
                                            button_width, row_y +
                                            45), (100, 100, 100), 1)
                text_x = button_x + 15
                cv2.putText(frame, button_text, (text_x, row_y + 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

                # Store button coordinates for mouse interaction
                param["button_coords"] = (
                    button_x, row_y + 15, button_x + button_width, row_y + 45)

            else:  # Numeric parameters with sliders
                # Draw slider track
                cv2.rectangle(frame, (slider_x_start, slider_y - 5),
                              (slider_x_end, slider_y + 5), (100, 100, 100), -1)

                # Calculate slider position
                value_range = param["max"] - param["min"]
                if value_range == 0:  # Avoid division by zero
                    value_range = 1
                slider_pos = int(
                    slider_x_start + (param["value"] - param["min"]) / value_range * slider_width)

                # Draw slider handle
                handle_radius = 8
                cv2.circle(frame, (slider_pos, slider_y),
                           handle_radius, (0, 200, 200), -1)
                cv2.circle(frame, (slider_pos, slider_y),
                           handle_radius, (200, 200, 200), 1)

                # Store slider coordinates for mouse interaction
                param["slider_coords"] = (
                    slider_x_start, slider_y - 10, slider_x_end, slider_y + 10)
                param["slider_pos"] = slider_pos

                # Get value with correct type display
                if name in ["Frame Skip"]:
                    value_str = str(int(param["value"]))
                else:
                    value_str = f"{param['value']:.2f}"

                # Parameter value
                cv2.putText(
                    frame,
                    value_str,
                    (slider_x_end + 10,
                     slider_y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255,
                     255,
                     255),
                    1)

#############################
# 7. MAIN FUNCTION
#############################


def draw_vehicle_trails(
        frame,
        current_time,
        width_scale,
        height_scale,
        crop_area=None):
    """Draw the vehicle trail history"""
    # Process each vehicle's trail
    for track_id, trail_points in list(vehicle_trail_history.items()):
        if len(trail_points) < 2:
            continue

        # Draw the trail as a polyline
        line_points = []
        for point in trail_points:
            cx, cy = point

            # Adjust for crop if needed
            if crop_area:
                crop_x1, crop_y1 = crop_area[0], crop_area[1]
                disp_x = int((cx - crop_x1) * width_scale)
                disp_y = int((cy - crop_y1) * height_scale)
            else:
                disp_x = int(cx * width_scale)
                disp_y = int(cy * height_scale)

            line_points.append((disp_x, disp_y))

        # Use a consistent color for each vehicle by using track_id as seed
        # Ensure seed is within valid range (0 to 2^32 - 1)
        seed_value = (hash(str(track_id)) & 0xFFFFFFFF)  # Limit to 32 bits
        np.random.seed(seed_value)
        color = (
            int(np.random.randint(100, 255)),
            int(np.random.randint(100, 255)),
            int(np.random.randint(100, 255))
        )

        # Convert to int array for polylines
        points_array = np.array(line_points, dtype=np.int32)
        cv2.polylines(frame, [points_array], False, color, 2)


def update_vehicle_trail(
        track_id,
        center_point,
        current_frame,
        zone_transition=False,
        current_zone=None,
        in_main_zone=True):
    """
    Update the vehicle's trail history based on its position, zone, and main zone status

    Parameters:
    - track_id: Unique identifier for the vehicle
    - center_point: Current position (x,y)
    - current_frame: Current frame number
    - zone_transition: Whether a zone transition occurred
    - current_zone: The current zone the vehicle is in (None if not in a specific zone)
    - in_main_zone: Whether the vehicle is in the main zone
    """
    # Start a trail when vehicle is in a specific zone (not just main zone)
    if current_zone is not None and current_zone != "MAIN":
        # Initialize trail if not exists
        if track_id not in vehicle_trail_history:
            vehicle_trail_history[track_id] = []

        # Add current position to the trail
        vehicle_trail_history[track_id].append(
            (center_point.x, center_point.y))

        # Limit trail length to avoid memory issues
        if len(vehicle_trail_history[track_id]) > MAX_TRAIL_LENGTH:
            vehicle_trail_history[track_id] = vehicle_trail_history[track_id][-MAX_TRAIL_LENGTH:]

    # Continue the trail if the vehicle is still in the main zone and already
    # has a trail
    elif in_main_zone and track_id in vehicle_trail_history:
        # Continue adding points to the trail as long as the vehicle is in the
        # main zone
        vehicle_trail_history[track_id].append(
            (center_point.x, center_point.y))

        # Limit trail length to avoid memory issues
        if len(vehicle_trail_history[track_id]) > MAX_TRAIL_LENGTH:
            vehicle_trail_history[track_id] = vehicle_trail_history[track_id][-MAX_TRAIL_LENGTH:]

    # If vehicle is no longer in the main zone, immediately remove its trail
    if not in_main_zone and track_id in vehicle_trail_history:
        # Remove the trail immediately instead of fading it out
        del vehicle_trail_history[track_id]
        if track_id in vehicle_arrived:
            del vehicle_arrived[track_id]
        if track_id in vehicle_completed:
            vehicle_completed.remove(track_id)

    # Handle zone transitions for existing trails
    if zone_transition and track_id in vehicle_trail_history:
        # When a vehicle transitions to a new zone, mark it for eventual trail removal
        # only if it's leaving the main zone completely
        if not in_main_zone and track_id not in vehicle_completed:
            vehicle_completed.add(track_id)
            # Immediately remove the trail when leaving main zone
            if track_id in vehicle_trail_history:
                del vehicle_trail_history[track_id]


def get_video_list():
    """Get the list of videos to process, either single video or from directory"""
    videos = []

    if VIDEOS_DIRECTORY:
        # Process all videos in the directory
        if not os.path.isdir(VIDEOS_DIRECTORY):
            print(f"Error: Directory {VIDEOS_DIRECTORY} does not exist")
            return videos

        # Get all video files from the directory
        for filename in os.listdir(VIDEOS_DIRECTORY):
            file_ext = os.path.splitext(filename)[1].lower()
            if file_ext in SUPPORTED_EXTENSIONS:
                videos.append(os.path.join(VIDEOS_DIRECTORY, filename))

        # Sort videos by name
        videos.sort()

        if not videos:
            print(f"No video files found in {VIDEOS_DIRECTORY}")
        else:
            print(f"Found {len(videos)} videos to process")
    elif VIDEO_PATH and os.path.exists(VIDEO_PATH):
        # Single video mode
        videos.append(VIDEO_PATH)
    else:
        print("No valid video path or directory specified")

    return videos


# Add a global dictionary to store persistent parameters
persistent_tracking_params = {}

def process_video(
        video_path,
        editor,
        first_video=False,
        last_frame_number=0,
        current_video_index=0,
        total_videos=1):
    """Process a single video file using the provided zone editor settings"""
    global vehicle_paths, last_known_zone, zone_transitions_count
    global vehicle_last_seen, vehicle_positions, vehicle_predictions
    global vehicle_trail_history, vehicle_arrived, vehicle_completed
    global time_segment_transitions, current_time_segment, last_segment_time
    global persistent_tracking_params
    
    # Helper function to safely destroy a window if it exists
    def safe_destroy_window(window_name):
        try:
            # getWindowProperty returns -1 if window doesn't exist
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) >= 0:
                cv2.destroyWindow(window_name)
        except:
            # If any error occurs, just pass (window might not exist)
            pass
    
    # Close any existing video windows
    safe_destroy_window("Parameters")
    safe_destroy_window("Tracking Summary")
    
    # Get the previous video window name if any
    if hasattr(process_video, 'previous_window') and process_video.previous_window:
        safe_destroy_window(process_video.previous_window)
    
    # Reset time segment tracking for new video
    time_segment_transitions = {}
    current_time_segment = 0
    last_segment_time = 0
    
    # Use the global checkpoint function directly
    def save_checkpoint_local(next_video_index, current_frame_number, video_count):
        """Save processing state by calling the global checkpoint function"""
        # Call the global function
        save_checkpoint(next_video_index, current_frame_number, video_count)
    
    # Only reset tracking data if this is the first video
    if first_video:
        vehicle_paths = {}
        last_known_zone = {}
        zone_transitions_count = {}
        vehicle_last_seen = {}
        vehicle_positions = {}
        vehicle_predictions = {}
        vehicle_trail_history = {}
        vehicle_arrived = {}
        vehicle_completed = set()

    # Get zones from the editor
    ZONES = editor.zones
    main_zone = editor.main_zone

    # Get crop area if set
    crop_area = None
    if editor.has_crop and editor.crop_start and editor.crop_end:
        crop_area = (editor.crop_start[0], editor.crop_start[1],
                     editor.crop_end[0], editor.crop_end[1])
        print(f"Using crop area: {crop_area}")

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: cannot open video {video_path}")
        return last_frame_number

    # Get video info
    video_name = os.path.basename(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_seconds = total_frames / fps if fps > 0 else 0

    # Create formatted duration string
    total_minutes = int(duration_seconds / 60)
    total_seconds = int(duration_seconds % 60)
    duration_str = f"{total_minutes:02d}:{total_seconds:02d}"

    print(
        f"Processing video: {video_name} (Duration: {duration_str}, {total_frames} frames)")

    # Load YOLO model
    model = YOLO(MODEL_PATH)
    model.conf = CONF_THRESHOLD
    model.iou = IOU_THRESHOLD

    # Create UI params with advanced options - use persistent parameters if available
    if persistent_tracking_params:
        tracking_params = persistent_tracking_params.copy()
    else:
        tracking_params = {
            "Confidence": {"value": CONF_THRESHOLD, "min": 0.1, "max": 1.0, "step": 0.05},
            "IOU": {"value": IOU_THRESHOLD, "min": 0.1, "max": 1.0, "step": 0.05},
            "Frame Skip": {"value": FRAME_SKIP, "min": 0, "max": 30, "step": 1},
            "Playback Speed": {"value": 10.0, "min": 0.1, "max": 10.0, "step": 0.1}, # Added playback speed
            "Show Labels": {"value": 1 if SHOW_LABELS else 0, "min": 0, "max": 1, "step": 1},
            "Show Zones": {"value": 1 if SHOW_ZONES else 0, "min": 0, "max": 1, "step": 1},
            "Show Tracks": {"value": 1 if SHOW_TRACKS else 0, "min": 0, "max": 1, "step": 1},
            "Show Trails": {"value": 0, "min": 0, "max": 1, "step": 1},
            "Enhanced Preproc": {"value": 1 if USE_ENHANCED_PREPROCESSING else 0, "min": 0, "max": 1, "step": 1},
            "_active_slider": None
        }

    # Add GPU info
    if torch.cuda.is_available():
        tracking_params["_gpu_info"] = torch.cuda.get_device_name(0)
    else:
        tracking_params["_gpu_info"] = "CPU"

    # Setup windows
    window_name = f"Vehicle Detection - {video_name}"
    cv2.namedWindow(window_name)
    cv2.moveWindow(window_name, 50, 50)  # Position main window
    
    # Store current window name for cleanup when starting next video
    process_video.previous_window = window_name

    # Create summary window
    summary_img = create_summary_window()
    cv2.moveWindow("Tracking Summary", 1000, 50)

    # Create parameter window
    params_img = create_parameter_window()
    cv2.moveWindow("Parameters", 550, 50)

    # Setup parameter window mouse callback
    def param_mouse_callback(event, x, y, flags, param):
        handle_param_mouse(event, x, y, flags, param, tracking_params)

    cv2.setMouseCallback("Parameters", param_mouse_callback)

    # Initialize frame accumulator if enabled
    frame_accumulator = FrameAccumulator(ACCUMULATION_FRAMES) if USE_FRAME_ACCUMULATION else None

    frame_count = 0
    processing_time = 0
    last_detection_boxes = []
    original_scale = None

    # Initialize tracking variables - continue from last frame number of
    # previous video
    frame_number = last_frame_number

    # For updating summary at regular intervals
    summary_update_interval = 5  # Update every 5 frames
    summary_counter = 0

    # For periodic checkpoint saving
    last_checkpoint_time = time.time()

    # For trail cleanup
    last_cleanup_frame = frame_number
    
    # For persistent display of diagnostics
    diag_preprocess_time = 0
    diag_detection_time = 0
    diag_gpu_mem = 0
    diag_update_counter = 0
    diag_update_interval = 15  # Update diagnostics every 15 frames to reduce flickering

    # Create results directory for saving output
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    # Process the video
    print(f"Starting processing for {video_name} (continuing from frame {frame_number})")

    # Store video start time for progress calculation
    vid_start_time = time.time()
    frames_processed = 0
    
    # Skip to the last processed frame if resuming
    current_frame = 0
    if last_frame_number > 0:
        print(f"Skipping to frame {last_frame_number}...")
        # Set video position to the last frame we processed
        # We can't directly set the frame position as some video codecs don't support it well
        # So we'll read frames until we reach the right position
        while current_frame < last_frame_number:
            ret = cap.read()[0]  # Just check if read was successful
            if not ret:
                print(f"Error: Could not skip to frame {last_frame_number}, video ended at {current_frame}")
                break
            current_frame += 1

    while True:
        start_time = time.time()
        ret, original_frame = cap.read()
        if not ret:
            break

        frames_processed += 1
        current_frame += 1

        # Store original frame dimensions for scaling back detection coordinates
        if original_scale is None:
            original_height, original_width = original_frame.shape[:2]

        # Apply crop if defined
        if crop_area:
            x1, y1, x2, y2 = crop_area
            # Ensure within image bounds
            x1 = max(0, min(x1, original_width - 1))
            y1 = max(0, min(y1, original_height - 1))
            x2 = max(0, min(x2, original_width))
            y2 = max(0, min(y2, original_height))

            # Crop the frame
            original_frame = original_frame[y1:y2, x1:x2]

            # Update dimension information after crop
            original_height, original_width = original_frame.shape[:2]

        # Resize frame for display
        frame = resize_frame(
            original_frame.copy(),
            MAX_DISPLAY_WIDTH,
            MAX_DISPLAY_HEIGHT)

        # Calculate scale factor between original and resized frame
        new_height, new_width = frame.shape[:2]
        width_scale = new_width / original_width
        height_scale = new_height / original_height

        frame_count += 1

        # Skip detection on some frames if needed, but always process for display
        detect_this_frame = frame_count % (int(tracking_params["Frame Skip"]["value"]) + 1) == 0

        # Always draw zones for stability (prevents flickering)
        if tracking_params["Show Zones"]["value"] == 1:
            # Existing zone drawing code...
            if main_zone is not None:
                coords = list(main_zone.exterior.coords)
                for i in range(len(coords) - 1):
                    # Adjust coordinates for crop if applicable
                    if crop_area:
                        crop_x1, crop_y1 = crop_area[0], crop_area[1]
                        pt1 = (int((coords[i][0] - crop_x1) * width_scale),
                               int((coords[i][1] - crop_y1) * height_scale))
                        pt2 = (int((coords[i + 1][0] - crop_x1) * width_scale),
                               int((coords[i + 1][1] - crop_y1) * height_scale))
                    else:
                        pt1 = (int(coords[i][0] * width_scale),
                               int(coords[i][1] * height_scale))
                        pt2 = (int(coords[i + 1][0] * width_scale),
                               int(coords[i + 1][1] * height_scale))
                    cv2.line(frame, pt1, pt2, (0, 0, 255), 2)

            for zone_name, zone_poly in ZONES.items():
                coords = list(zone_poly.exterior.coords)
                for i in range(len(coords) - 1):
                    # Adjust coordinates for crop if applicable
                    if crop_area:
                        crop_x1, crop_y1 = crop_area[0], crop_area[1]
                        pt1 = (int((coords[i][0] - crop_x1) * width_scale),
                               int((coords[i][1] - crop_y1) * height_scale))
                        pt2 = (int((coords[i + 1][0] - crop_x1) * width_scale),
                               int((coords[i + 1][1] - crop_y1) * height_scale))
                        label_x = int((coords[0][0] - crop_x1) * width_scale)
                        label_y = int((coords[0][1] - crop_y1) * height_scale) - 10
                    else:
                        pt1 = (int(coords[i][0] * width_scale),
                               int(coords[i][1] * height_scale))
                        pt2 = (int(coords[i + 1][0] * width_scale),
                               int(coords[i + 1][1] * height_scale))
                        label_x = int(coords[0][0] * width_scale)
                        label_y = int(coords[0][1] * height_scale) - 10

                    cv2.line(frame, pt1, pt2, (255, 0, 0), 2)

                    # Only draw label if the first point is visible in the cropped frame
                    if 0 <= label_x < frame.shape[1] and 0 <= label_y < frame.shape[0]:
                        cv2.putText(frame, zone_name, (label_x, label_y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        # Clean up old trails periodically to prevent memory issues
        if frame_number - last_cleanup_frame >= TRACK_CLEANUP_INTERVAL:
            # Clean up very old trails that are no longer needed
            current_time = time.time()
            for track_id in list(vehicle_trail_history.keys()):
                # If trail is too long, trim it
                if len(vehicle_trail_history[track_id]) > MAX_TRAIL_LENGTH:
                    vehicle_trail_history[track_id] = vehicle_trail_history[track_id][-MAX_TRAIL_LENGTH:]

                # If track was marked as arrived long ago, remove it completely
                if track_id in vehicle_arrived:
                    time_since_arrival = current_time - vehicle_arrived[track_id]
                    if time_since_arrival > TRAIL_FADE_SECONDS * 2:  # Double fade time for safety
                        del vehicle_trail_history[track_id]
                        vehicle_arrived.pop(track_id, None)

            last_cleanup_frame = frame_number
            print(f"Cleaned up trails. Active trails: {len(vehicle_trail_history)}")

        # Draw vehicle trails if enabled
        if tracking_params["Show Trails"]["value"] == 1:
            draw_vehicle_trails(frame, time.time(), width_scale, height_scale, crop_area)

        # Draw previous detections (always, prevents flickering)
        if tracking_params["Show Tracks"]["value"] == 1 and last_detection_boxes:
            for box_data in last_detection_boxes:
                x1, y1, x2, y2, cls_id, track_id = box_data

                # Scale box coordinates to match resized frame
                disp_x1, disp_y1 = x1 * width_scale, y1 * height_scale
                disp_x2, disp_y2 = x2 * width_scale, y2 * height_scale

                cv2.rectangle(frame, (int(disp_x1), int(disp_y1)),
                              (int(disp_x2), int(disp_y2)), (0, 255, 0), 2)

                if tracking_params["Show Labels"]["value"] == 1:
                    cls_names = [
                        "person",
                        "bicycle",
                        "car",
                        "motorcycle",
                        "airplane",
                        "bus",
                        "train",
                        "truck"]
                    label = cls_names[cls_id] if cls_id < len(
                        cls_names) else f"class:{cls_id}"
                    cv2.putText(frame, label, (int(disp_x1), int(disp_y1) - 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                if tracking_params["Show Tracks"]["value"] == 1:
                    cv2.putText(frame, f"ID {track_id}", (int(disp_x1), int(
                        disp_y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Only run detection on some frames based on Frame Skip
        if detect_this_frame:
            # Apply preprocessing based on GPU availability
            start_preprocess = time.time()
            if tracking_params["Enhanced Preproc"]["value"] == 1:
                processed_frame = enhanced_preprocess(original_frame)
            else:
                processed_frame = original_frame
            
            # Frame accumulation is disabled to avoid detection issues
            
            # Time the preprocessing for feedback
            preprocess_time = time.time() - start_preprocess
            
            # Update model parameters from UI controls
            model.conf = tracking_params["Confidence"]["value"]
            model.iou = tracking_params["IOU"]["value"]
            
            # Explicitly ensure we're using the GPU for inference
            if USE_GPU and torch.cuda.is_available():
                torch.cuda.empty_cache()  # Clear GPU memory
            
            # Increment frame counter
            frame_number += 1
            
            # Perform detection with timing
            start_detection = time.time()
            
            # Multi-scale detection is disabled to avoid issues
            # Standard detection with tracking
            results = model.track(
                source=processed_frame,
                persist=True,
                conf=tracking_params["Confidence"]["value"],
                iou=tracking_params["IOU"]["value"],
                imgsz=IMG_SIZE,
                device=DEVICE,
                half=HALF_PRECISION,
                tracker=TRACKER,
                verbose=False,
                retina_masks=False,
                classes=[2, 3, 5, 7]  # Only car, motorcycle, bus, truck
            )
            
            # Calculate and display timing info
            detection_time = time.time() - start_detection
            
            # Update diagnostic values with smoothing
            diag_update_counter += 1
            if diag_update_counter >= diag_update_interval:
                diag_preprocess_time = preprocess_time
                diag_detection_time = detection_time
                if torch.cuda.is_available():
                    diag_gpu_mem = torch.cuda.memory_allocated() / 1024 / 1024  # MB
                diag_update_counter = 0
            
            # Store currently detected vehicle IDs
            current_track_ids = set()
            last_detection_boxes = []

            # Process results
            if len(results) > 0:
                tracked_result = results[0]
                boxes = tracked_result.boxes
                if boxes is not None:
                    for box in boxes:
                        if box.id is None:
                            continue
                        track_id = int(box.id[0])
                        current_track_ids.add(track_id)

                        cls_id = int(box.cls[0])
                        # Filter vehicles (car, motorcycle, bus, truck)
                        if cls_id not in [2, 3, 5, 7]:
                            continue

                        # Get coordinates in original frame
                        x1, y1, x2, y2 = box.xyxy[0].tolist()

                        # Filter by area and aspect-ratio to kill obvious false positives
                        box_w = (x2 - x1)
                        box_h = (y2 - y1)
                        box_ar = box_w / (box_h + 1e-6)
                        box_area = box_w * box_h
                        frame_area = original_width * original_height
                        
                        # More permissive filtering for low-quality videos
                        if box_area < 0.0002 * frame_area or box_area > 0.5 * frame_area:
                            continue  # Too small or too large
                        if box_ar < 0.3 or box_ar > 5.0:  # More permissive aspect ratio filtering
                            continue

                        # Center point for zone checking
                        orig_cx = (x1 + x2) / 2.0
                        orig_cy = (y1 + y2) / 2.0

                        # Adjust center point coordinates if using crop
                        if crop_area:
                            crop_x1, crop_y1 = crop_area[0], crop_area[1]
                            # Add crop offset for zone checking
                            center_point = Point(
                                orig_cx + crop_x1, orig_cy + crop_y1)
                        else:
                            center_point = Point(orig_cx, orig_cy)

                        # Skip if not in main zone
                        if main_zone is not None and not main_zone.contains(
                                center_point):
                            continue

                        # Update vehicle tracking data
                        vehicle_last_seen[track_id] = frame_number
                        vehicle_positions[track_id] = (
                            orig_cx, orig_cy, x1, y1, x2, y2)

                        # Only track vehicles in the main zone
                        last_detection_boxes.append(
                            (x1, y1, x2, y2, cls_id, track_id))

                        # Scale box coordinates for display
                        disp_x1, disp_y1 = x1 * width_scale, y1 * height_scale
                        disp_x2, disp_y2 = x2 * width_scale, y2 * height_scale

                        # Draw bounding box on display frame
                        cv2.rectangle(frame, (int(disp_x1), int(disp_y1)), (int(
                            disp_x2), int(disp_y2)), (0, 255, 0), 2)

                        if tracking_params["Show Labels"]["value"] == 1:
                            cls_names = [
                                "person",
                                "bicycle",
                                "car",
                                "motorcycle",
                                "airplane",
                                "bus",
                                "train",
                                "truck"]
                            label = cls_names[cls_id] if cls_id < len(
                                cls_names) else f"class:{cls_id}"
                            cv2.putText(frame, label, (int(disp_x1), int(disp_y1) - 30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                        if tracking_params["Show Tracks"]["value"] == 1:
                            cv2.putText(frame, f"ID {track_id}", (int(disp_x1), int(
                                disp_y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                        # Check which zone contains the vehicle
                        current_zone = None
                        for zone_name, zone_poly in ZONES.items():
                            if zone_poly.contains(center_point):
                                current_zone = zone_name
                                break

                        # Check if vehicle is in main zone
                        in_main_zone = main_zone is not None and main_zone.contains(
                            center_point)

                        # Track zone transitions
                        zone_transition_detected = False

                        if current_zone is not None:
                            if track_id not in last_known_zone:
                                # First time seeing this vehicle in a zone
                                last_known_zone[track_id] = current_zone
                                vehicle_paths[track_id] = [current_zone]
                            else:
                                # Vehicle already had a zone
                                prev_zone = last_known_zone[track_id]
                                if prev_zone != current_zone:
                                    # Zone transition detected
                                    transition_key = (prev_zone, current_zone)
                                    zone_transitions_count[transition_key] = (
                                        zone_transitions_count.get(transition_key, 0) + 1
                                    )
                                    
                                    # Also track for time segment
                                    time_segment_transitions[transition_key] = (
                                        time_segment_transitions.get(transition_key, 0) + 1
                                    )
                                    
                                    vehicle_paths[track_id].append(current_zone)
                                    last_known_zone[track_id] = current_zone
                                    zone_transition_detected = True
                                    
                                    # Print transition for debugging
                                    print(f"Vehicle {track_id} moved from {prev_zone} to {current_zone}")

                        # Update vehicle trail with current position and zone information
                        update_vehicle_trail(
                            track_id,
                            center_point,
                            frame_number,
                            zone_transition_detected,
                            current_zone,
                            in_main_zone
                        )

            # Handle temporarily occluded vehicles
            occluded_track_ids = set()
            for track_id, last_frame in list(vehicle_last_seen.items()):
                # Skip vehicles that were just detected
                if track_id in current_track_ids:
                    continue

                # Only process for a limited number of frames after the vehicle was last seen
                frames_since_seen = frame_number - last_frame
                if frames_since_seen <= TRACK_BUFFER:  # Use the enhanced TRACK_BUFFER
                    # If we have position data for this vehicle
                    if track_id in vehicle_positions:
                        occluded_track_ids.add(track_id)
                        
                        # Don't flicker - only show prediction boxes for vehicles that have been seen recently
                        # and weren't just created (to filter out unstable tracks)
                        if frames_since_seen < MAX_TRACK_AGE and frames_since_seen > 4:
                            cx, cy, x1, y1, x2, y2 = vehicle_positions[track_id]

                            # Draw predicted box with dashed lines
                            disp_x1, disp_y1 = x1 * width_scale, y1 * height_scale
                            disp_x2, disp_y2 = x2 * width_scale, y2 * height_scale

                            # Draw with orange color and reduced opacity for predicted positions
                            alpha = max(0.3, 1.0 - (frames_since_seen / TRACK_BUFFER))
                            overlay_box = frame.copy()
                            cv2.rectangle(overlay_box, (int(disp_x1), int(disp_y1)), (int(
                                disp_x2), int(disp_y2)), (0, 165, 255), 2)

                            # Apply transparency
                            cv2.addWeighted(
                                overlay_box, alpha, frame, 1 - alpha, 0, frame)

                            if tracking_params["Show Tracks"]["value"] == 1:
                                cv2.putText(frame, f"ID {track_id} (pred)",
                                            (int(disp_x1), int(disp_y1) - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
                elif frames_since_seen > MAX_TRACK_AGE:
                    # Remove very old tracks
                    vehicle_last_seen.pop(track_id, None)
                    vehicle_positions.pop(track_id, None)
                    vehicle_predictions.pop(track_id, None)

        # Draw diagnostic information on every frame but update values less frequently
        # Create semi-transparent background for diagnostics
        overlay = frame.copy()
        # Draw black rectangle for better readability
        cv2.rectangle(overlay, (5, 125), (220, 220), (0, 0, 0), -1)
        # Apply the overlay with transparency
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Draw the diagnostic text
        cv2.putText(frame, f"Preprocess: {diag_preprocess_time:.3f}s", (10, 150),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, f"Detect: {diag_detection_time:.3f}s", (10, 180),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        if torch.cuda.is_available():
            cv2.putText(frame, f"GPU Mem: {diag_gpu_mem:.1f}MB", (10, 210),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
        # Update summary window periodically
        summary_counter += 1

        # Calculate video progress
        if fps > 0:
            processed_seconds = frames_processed / fps
            minutes = int(processed_seconds / 60)
            seconds = int(processed_seconds % 60)
            total_minutes = int(duration_seconds / 60)
            total_seconds = int(duration_seconds % 60)

            # Check if we've reached a new time segment (every 10 minutes)
            current_segment = minutes // time_interval_minutes
            if current_segment > current_time_segment:
                # We've entered a new time segment, save the previous one
                start_minute = current_time_segment * time_interval_minutes
                end_minute = start_minute + time_interval_minutes - 1
                
                # Save the summary for this time segment
                video_name = os.path.basename(video_path)
                save_time_segment_summary(video_name, start_minute, end_minute, 
                                         time_segment_transitions, results_dir)
                
                # Reset for the new segment
                current_time_segment = current_segment
                time_segment_transitions = {}
                
            # Format as MM:SS
            time_str = f"{minutes:02d}:{seconds:02d} / {total_minutes:02d}:{total_seconds:02d}"

            # Calculate percentage complete
            percent = min(
                100, int(
                    (frames_processed / total_frames) * 100)) if total_frames > 0 else 0

            # Video progress overlay
            progress_text = f"Video {current_video_index+1}/{total_videos} | {time_str} ({percent}%)"

            # Add transparent background for better visibility
            text_size = cv2.getTextSize(
                progress_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            overlay = frame.copy()
            cv2.rectangle(overlay, (10, 60), (10 +
                                              text_size[0] +
                                              10, 60 +
                                              text_size[1] +
                                              10), (0, 0, 0), -
                          1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
            cv2.putText(frame, progress_text, (15, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Show FPS
        processing_time = time.time() - start_time
        fps_text = f"FPS: {1.0/processing_time:.1f}"
        cv2.putText(frame, fps_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Update parameter window
        draw_param_ui(params_img, tracking_params, separate_window=True)

        # Update summary window periodically
        summary_counter += 1
        if summary_counter >= summary_update_interval:
            summary_counter = 0
            update_summary_window(
                summary_img,
                zone_transitions_count,
                vehicle_paths,
                ZONES)

        # Display the frame
        cv2.imshow(window_name, frame)
        
        # Use playback speed to control wait time
        wait_time = int(max(1, 30 / tracking_params["Playback Speed"]["value"]))
        key = cv2.waitKey(wait_time) & 0xFF

        # Save checkpoint periodically
        current_time = time.time()
        if current_time - last_checkpoint_time >= SAVE_CHECKPOINT_INTERVAL:
            save_checkpoint_local(current_video_index, frame_number, total_videos)
            last_checkpoint_time = current_time
            print(f"Auto-saved checkpoint at frame {frame_number}")

        if key == 27:  # ESC key
            cap.release()
            print("Processing aborted by user")
            return False  # Return False specifically for abort case
        elif key == ord('c'):  # 'c' key for manual checkpoint
            save_checkpoint_local(current_video_index, frame_number, total_videos)
            print(f"Manually saved checkpoint at frame {frame_number}")
    
    # At the end of process_video function, before returning
    # Save the last time segment if it has any transitions
    if time_segment_transitions:
        start_minute = current_time_segment * time_interval_minutes
        # Calculate the end minute based on actual processed video time
        if fps > 0:
            total_processed_minutes = int(frames_processed / fps / 60)
            end_minute = min(start_minute + time_interval_minutes - 1, total_processed_minutes)
        else:
            end_minute = start_minute + time_interval_minutes - 1
            
        # Save the final time segment summary
        video_name = os.path.basename(video_path)
        save_time_segment_summary(video_name, start_minute, end_minute, 
                                 time_segment_transitions, results_dir)
    
    # Store the current parameters for the next video
    # Make a deep copy of tracking_params, excluding internal variables
    persistent_tracking_params = {k: v.copy() if isinstance(v, dict) else v 
                                  for k, v in tracking_params.items() 
                                  if not k.startswith('_')}
    
    cap.release()

    # Save the results
    results_filename = os.path.join(
        results_dir, f"{os.path.splitext(video_name)[0]}_results.json")
    save_results(results_filename, zone_transitions_count, vehicle_paths)

    print(f"Finished processing {video_name}")
    print(f"Results saved to {results_filename}")

    # Print summary
    print("\n=== Zone Transition Counts ===")
    for (from_zone, to_zone), count in zone_transitions_count.items():
        print(f"{from_zone} -> {to_zone}: {count}")

    print("\n=== Vehicle Paths ===")
    for track_id, path in vehicle_paths.items():
        print(f"Vehicle {track_id} path: {path}")

    # Wait a moment to view results
    cv2.waitKey(1000)
    return frame_number


def save_results(filename, zone_transitions, vehicle_paths):
    """Save the tracking results to a JSON file"""
    results = {
        "zone_transitions": {
            f"{from_zone}->{to_zone}": count for (
                from_zone,
                to_zone),
            count in zone_transitions.items()},
        "vehicle_paths": {
            str(track_id): path for track_id,
            path in vehicle_paths.items()}}

    with open(filename, 'w') as f:
        json.dump(results, f, indent=4)


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Vehicle Tracking System")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint")
    args = parser.parse_args()

    # GPU detection
    if torch.cuda.is_available() and USE_GPU:
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    else:
        print("Using CPU for inference")

    # Get list of videos first
    videos = get_video_list()
    if not videos:
        print("No videos to process. Exiting.")
        return

    # Variables to track processing state
    start_video_idx = 0
    last_frame_number = 0
    
    # If not resuming, make sure to clear any existing checkpoint
    if not args.resume and os.path.exists(CHECKPOINT_FILE):
        print("Not resuming - removing existing checkpoint file for a fresh start")
        try:
            os.remove(CHECKPOINT_FILE)
        except Exception as e:
            print(f"Warning: Could not remove checkpoint file: {e}")

    # Check if we should resume from checkpoint
    if args.resume:
        checkpoint_data = load_checkpoint()
        if checkpoint_data:
            start_video_idx = checkpoint_data["next_video_index"]
            last_frame_number = checkpoint_data["last_frame_number"]
            restore_from_checkpoint(checkpoint_data)
            print(
                f"Resuming from video #{start_video_idx+1} (frame {last_frame_number})")
        else:
            print("No checkpoint found. Starting from the beginning.")

    # Run zone editor to set up zones (only needs to be done once with the
    # first video)
    editor_video = videos[0]  # Always use first video for zone setup
    editor = ZoneEditor(editor_video)
    if not editor.run():
        print("Zone editor was closed without continuing. Exiting.")
        return

    # Create dummy local function to match process_video's local function
    def save_checkpoint_for_main(next_video_index, frame_number, video_count):
        # Using the globally defined checkpoint functions
        save_checkpoint(next_video_index, frame_number, video_count)

    # Process each video sequentially, maintaining continuity
    for i, video_path in enumerate(
            videos[start_video_idx:], start=start_video_idx):
        # Only reset tracking for the first video if not resuming
        first_video = (i == 0 and not args.resume)

        print(
            f"Processing video {i+1}/{len(videos)}: {os.path.basename(video_path)}")
        last_frame_number = process_video(
            video_path,
            editor,
            first_video,
            last_frame_number if i == start_video_idx else 0, # Only use last_frame_number for the first video
            current_video_index=i,
            total_videos=len(videos)
        )

        # Check if processing was aborted
        if last_frame_number is False:
            print(f"Processing stopped at {video_path}")
            # Save checkpoint on abort
            save_checkpoint_for_main(i, last_frame_number, len(videos))
            break

        # Save checkpoint after each video
        save_checkpoint_for_main(i + 1, last_frame_number, len(videos))
        print(
            f"Completed video {i+1}/{len(videos)} with last frame number: {last_frame_number}")

    # Save final combined results
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    results_filename = os.path.join(results_dir, "combined_results.json")
    save_results(results_filename, zone_transitions_count, vehicle_paths)
    print(f"Saved combined results to {results_filename}")

    # Clean up all windows
    cv2.destroyAllWindows()
    print("All videos processed.")


if __name__ == "__main__":
    main()
