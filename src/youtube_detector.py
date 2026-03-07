# -*- coding: utf-8 -*-
"""
youtube_detector.py
--------------------
Smart Traffic Management System - Guntur & Vijayawada, AP
Real-Time Vehicle Detector using YOLOv8

FIXES APPLIED:
  - Confidence lowered to 0.15  (works for aerial/top-down videos)
  - Uses YOLOv8s model           (more accurate than nano)
  - Arrow symbols removed        (fixes Windows encoding issue)
  - Frame skipping added         (faster on CPU)
  - Video loops automatically    (no need to restart)
  - Better on-screen display

USAGE:
  # Video file on laptop:
  python src/youtube_detector.py --url "C:\\Users\\YourName\\Desktop\\traffic.mp4" --lane 1

  # Webcam:
  python src/youtube_detector.py --lane 1

  # YouTube video:
  python src/youtube_detector.py --url "https://www.youtube.com/watch?v=XXXX" --lane 1

INSTALL:
  pip install ultralytics opencv-python yt-dlp numpy
"""

import cv2
import sqlite3
import argparse
import time
import sys
from datetime import datetime

# ── Try importing YOLO ────────────────────────────────────────────────────
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("[ERROR] ultralytics not installed. Run: pip install ultralytics")

# ── YOLO COCO class IDs that count as vehicles ────────────────────────────
VEHICLE_CLASSES = {
    1:  "bike",    # bicycle
    2:  "car",     # car
    3:  "bike",    # motorbike (2-wheeler - most common in AP)
    5:  "bus",     # bus (APSRTC)
    7:  "truck",   # truck / lorry
}

# ── AP Lane names (plain text - no special characters for Windows) ─────────
AP_LANES = {
    1: "Brodipet - Amaravati Rd",
    2: "Benz Circle - MG Road",
    3: "NH-16 - Vijayawada",
    4: "Lakshmipuram - Besant Rd",
}

DB_PATH        = "traffic.db"
CONFIDENCE     = 0.15   # LOW = detects vehicles in aerial/top-down footage
MODEL_NAME     = "yolov8s.pt"  # small model - better than nano for accuracy
FRAME_SKIP     = 2      # process every 2nd frame (speeds up CPU)
SAVE_INTERVAL  = 1.0    # save counts to DB every 1 second

# ── DB Setup ──────────────────────────────────────────────────────────────
def init_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS vehicle_count (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            lane             INTEGER,
            lane_name        TEXT,
            timestamp        TEXT,
            cars             INTEGER,
            bikes            INTEGER,
            buses            INTEGER,
            trucks           INTEGER,
            total            INTEGER,
            emergency        INTEGER DEFAULT 0,
            congestion_level TEXT,
            signal_state     TEXT    DEFAULT 'PENDING',
            green_duration   INTEGER DEFAULT 0,
            wait_time        INTEGER DEFAULT 0,
            reason           TEXT    DEFAULT '',
            priority_score   INTEGER DEFAULT 0
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS signal_log (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp      TEXT,
            lane           INTEGER,
            lane_name      TEXT,
            green_duration INTEGER,
            reason         TEXT
        )
    """)
    conn.commit()
    return conn

def get_congestion(total):
    if total == 0:  return "CLEAR"
    if total <= 5:  return "LOW"
    if total <= 12: return "MEDIUM"
    if total <= 20: return "HIGH"
    return "CRITICAL"

def save_counts(conn, lane_id, lane_name, cars, bikes, buses, trucks, emergency):
    total = cars + bikes + buses + trucks
    ts    = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn.execute("""
        INSERT INTO vehicle_count
        (lane, lane_name, timestamp, cars, bikes, buses, trucks,
         total, emergency, congestion_level)
        VALUES (?,?,?,?,?,?,?,?,?,?)
    """, (lane_id, lane_name, ts,
          cars, bikes, buses, trucks,
          total, 1 if emergency else 0,
          get_congestion(total)))
    conn.commit()

# ── Resolve YouTube URL ───────────────────────────────────────────────────
def resolve_youtube_url(url):
    try:
        import yt_dlp
        print("[INFO] Resolving YouTube URL with yt-dlp...")
        ydl_opts = {
            "quiet":      True,
            "format":     "best[ext=mp4][height<=720]/best[ext=mp4]/best",
            "noplaylist": True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info       = ydl.extract_info(url, download=False)
            stream_url = info.get("url") or info["formats"][-1]["url"]
            title      = info.get("title", "Unknown")
            print(f"[OK]   Stream: {title}")
            return stream_url
    except ImportError:
        print("[WARN] yt-dlp not found. Run: pip install yt-dlp")
        return url
    except Exception as e:
        print(f"[WARN] Could not resolve YouTube URL: {e}")
        return url

# ── Emergency detection heuristic ────────────────────────────────────────
def is_emergency(x1, y1, x2, y2, frame_h, frame_w, conf):
    """
    Flags a vehicle as emergency if it is very large in frame AND high confidence.
    In real deployment: use GPS transponder or siren audio detection.
    """
    vehicle_h = y2 - y1
    vehicle_w = x2 - x1
    if (vehicle_h / frame_h > 0.40 or vehicle_w / frame_w > 0.35) and conf > 0.75:
        return True
    return False

# ── Draw info overlay on OpenCV window ───────────────────────────────────
def draw_overlay(frame, lane_id, lane_name, counts, congestion, is_emg):
    h, w = frame.shape[:2]
    total = counts["cars"] + counts["bikes"] + counts["buses"] + counts["trucks"]

    # Top black bar
    cv2.rectangle(frame, (0, 0), (w, 80), (0, 0, 0), -1)

    # Lane name
    cv2.putText(frame,
                f"Lane {lane_id}: {lane_name}",
                (10, 25), cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 255, 255), 2)

    # Vehicle counts
    cv2.putText(frame,
                f"Cars:{counts['cars']}  Bikes:{counts['bikes']}  Buses:{counts['buses']}  Trucks:{counts['trucks']}",
                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 220, 255), 1)

    # Total + congestion level
    cong_colors = {
        "CLEAR":    (0, 255, 200),
        "LOW":      (0, 255, 100),
        "MEDIUM":   (0, 200, 255),
        "HIGH":     (0, 140, 255),
        "CRITICAL": (0, 60, 255),
    }
    color = cong_colors.get(congestion, (255, 255, 255))
    cv2.putText(frame,
                f"Total: {total}  |  {congestion}",
                (10, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)

    # Emergency banner
    if is_emg:
        cv2.rectangle(frame, (0, 80), (w, 115), (0, 0, 180), -1)
        cv2.putText(frame,
                    "  EMERGENCY VEHICLE DETECTED - PRIORITY GREEN",
                    (10, 105), cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 255, 255), 2)

    # Dashboard reminder bottom
    cv2.putText(frame,
                "Dashboard: http://localhost:5000",
                (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 100, 100), 1)

    return frame

# ── MAIN DETECTION LOOP ───────────────────────────────────────────────────
def run(url, lane_id, show_window, conf_thresh, model_name):
    if not YOLO_AVAILABLE:
        print("[ERROR] ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)

    lane_name = AP_LANES.get(lane_id, f"Lane {lane_id}")

    print("")
    print("=" * 58)
    print("  Smart Traffic Management System")
    print("  Guntur & Vijayawada, Andhra Pradesh - SIH 2025")
    print("=" * 58)
    print(f"  Lane      : {lane_id} - {lane_name}")
    print(f"  Source    : {url if url else 'Webcam (camera 0)'}")
    print(f"  Model     : {model_name}")
    print(f"  Confidence: {conf_thresh}")
    print(f"  DB        : {DB_PATH}")
    print(f"  Dashboard : http://localhost:5000")
    print("=" * 58)
    print("")

    # Load YOLO model
    print(f"[INFO] Loading {model_name} (downloads ~22MB on first run)...")
    model = YOLO(model_name)
    print(f"[OK]   Model loaded.\n")

    # Open video source
    if url:
        if "youtube.com" in url or "youtu.be" in url:
            stream_url = resolve_youtube_url(url)
        else:
            stream_url = url
    else:
        stream_url = 0

    print(f"[INFO] Opening video source...")
    cap = cv2.VideoCapture(stream_url)

    if not cap.isOpened():
        print(f"\n[ERROR] Could not open: {stream_url}")
        print("\n  Possible fixes:")
        print("  1. Check the file path is correct (use double backslashes on Windows)")
        print("     Example: C:\\\\Users\\\\YourName\\\\Desktop\\\\traffic.mp4")
        print("  2. Drag and drop the file into this prompt to get the exact path")
        print("  3. Make sure the file is not open in another program")
        sys.exit(1)

    # Video info
    fps    = cap.get(cv2.CAP_PROP_FPS) or 25
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[OK]   Video opened: {width}x{height} @ {fps:.0f} fps")
    print(f"[INFO] Press Q in the video window to stop.\n")

    # Init DB and counters
    conn        = init_db()
    frame_count = 0
    last_save   = time.time()
    roll        = {"cars": 0, "bikes": 0, "buses": 0, "trucks": 0, "emergency": False}
    roll_frames = 0
    last_counts    = {"cars": 0, "bikes": 0, "buses": 0, "trucks": 0}
    last_congestion = "CLEAR"
    last_emg        = False

    try:
        while True:
            ret, frame = cap.read()

            # Handle end of file - loop video
            if not ret:
                if url and "youtube" not in str(url) and "rtsp" not in str(url):
                    print("[INFO] Video ended - restarting from beginning...")
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    print("[WARN] Stream lost. Reconnecting in 3s...")
                    time.sleep(3)
                    cap = cv2.VideoCapture(stream_url)
                    continue

            frame_count += 1

            # Skip frames for CPU speed
            if frame_count % FRAME_SKIP != 0:
                if show_window:
                    display = draw_overlay(
                        frame.copy(), lane_id, lane_name,
                        last_counts, last_congestion, last_emg
                    )
                    cv2.imshow(f"Smart Traffic - Lane {lane_id}", display)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                continue

            h, w = frame.shape[:2]

            # Run YOLO
            results = model(frame, verbose=False, conf=conf_thresh)

            # Count vehicles this frame
            frame_counts = {"cars": 0, "bikes": 0, "buses": 0, "trucks": 0, "emg": False}
            for box_data in results[0].boxes:
                cls_id = int(box_data.cls[0])
                conf   = float(box_data.conf[0])
                if cls_id not in VEHICLE_CLASSES:
                    continue
                x1, y1, x2, y2 = map(int, box_data.xyxy[0])
                vtype = VEHICLE_CLASSES[cls_id]
                if vtype == "car":    frame_counts["cars"]   += 1
                elif vtype == "bike": frame_counts["bikes"]  += 1
                elif vtype == "bus":  frame_counts["buses"]  += 1
                elif vtype == "truck":frame_counts["trucks"] += 1
                if is_emergency(x1, y1, x2, y2, h, w, conf):
                    frame_counts["emg"] = True

            # Accumulate rolling window
            roll["cars"]   += frame_counts["cars"]
            roll["bikes"]  += frame_counts["bikes"]
            roll["buses"]  += frame_counts["buses"]
            roll["trucks"] += frame_counts["trucks"]
            if frame_counts["emg"]:
                roll["emergency"] = True
            roll_frames += 1

            # Update display cache
            last_counts = {
                "cars":   frame_counts["cars"],
                "bikes":  frame_counts["bikes"],
                "buses":  frame_counts["buses"],
                "trucks": frame_counts["trucks"],
            }
            last_congestion = get_congestion(sum(last_counts.values()))
            last_emg        = frame_counts["emg"]

            # Save to DB every second
            now = time.time()
            if now - last_save >= SAVE_INTERVAL and roll_frames > 0:
                avg_cars   = round(roll["cars"]   / roll_frames)
                avg_bikes  = round(roll["bikes"]  / roll_frames)
                avg_buses  = round(roll["buses"]  / roll_frames)
                avg_trucks = round(roll["trucks"] / roll_frames)
                emg        = roll["emergency"]

                save_counts(conn, lane_id, lane_name,
                            avg_cars, avg_bikes, avg_buses, avg_trucks, emg)

                total   = avg_cars + avg_bikes + avg_buses + avg_trucks
                emg_str = "  <<< EMERGENCY" if emg else ""
                print(
                    f"[{datetime.now().strftime('%H:%M:%S')}] "
                    f"Lane {lane_id} | "
                    f"Cars:{avg_cars:2d}  Bikes:{avg_bikes:2d}  "
                    f"Buses:{avg_buses:2d}  Trucks:{avg_trucks:2d}  "
                    f"Total:{total:2d}  {get_congestion(total):8s}"
                    f"{emg_str}"
                )

                roll        = {"cars": 0, "bikes": 0, "buses": 0, "trucks": 0, "emergency": False}
                roll_frames = 0
                last_save   = now

            # Show annotated window
            if show_window:
                annotated = results[0].plot()
                display   = draw_overlay(
                    annotated, lane_id, lane_name,
                    last_counts, last_congestion, last_emg
                )
                cv2.imshow(f"Smart Traffic - Lane {lane_id}", display)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user (Ctrl+C).")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        conn.close()
        print("[OK]   Detector stopped cleanly.")

# ── ENTRY POINT ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Smart Traffic - Real-Time Detector (Guntur/Vijayawada AP)",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--url", type=str, default=None,
        help=(
            "Video source. Examples:\n"
            "  Local file : C:\\Users\\Name\\Desktop\\traffic.mp4\n"
            "  YouTube    : https://www.youtube.com/watch?v=XXXX\n"
            "  IP Camera  : rtsp://192.168.1.10:554/stream\n"
            "  (blank)    : uses laptop webcam"
        )
    )
    parser.add_argument(
        "--lane", type=int, default=1, choices=[1, 2, 3, 4],
        help=(
            "Junction lane this camera covers:\n"
            "  1 = Brodipet - Amaravati Rd\n"
            "  2 = Benz Circle - MG Road\n"
            "  3 = NH-16 - Vijayawada\n"
            "  4 = Lakshmipuram - Besant Rd"
        )
    )
    parser.add_argument(
        "--no-window", action="store_true",
        help="Run without display window (headless mode)"
    )
    parser.add_argument(
        "--conf", type=float, default=CONFIDENCE,
        help=f"Detection confidence (default: {CONFIDENCE}). Lower = detects more vehicles."
    )
    parser.add_argument(
        "--model", type=str, default=MODEL_NAME,
        choices=["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"],
        help="Model size: n=fastest, s=balanced, m=most accurate (default: yolov8s.pt)"
    )
    args = parser.parse_args()
    run(
        url         = args.url,
        lane_id     = args.lane,
        show_window = not args.no_window,
        conf_thresh = args.conf,
        model_name  = args.model,
    )
