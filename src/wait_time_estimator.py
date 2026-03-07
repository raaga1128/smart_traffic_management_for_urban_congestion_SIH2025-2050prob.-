# -*- coding: utf-8 -*-
"""
wait_time_estimator.py
-----------------------
Smart Traffic Management System - Guntur & Vijayawada, AP

Estimates the WAITING TIME for vehicles at a junction signal
based on the count of vehicles detected from an image or video.

HOW WAITING TIME IS CALCULATED:
  Each vehicle type has a different "saturation flow rate" —
  how many vehicles of that type can clear the junction per second.

  Saturation flow rates (vehicles per second):
    Bike / 2-wheeler / Auto : 0.50  (fast, nimble)
    Car                     : 0.38  (medium)
    Bus (APSRTC)            : 0.22  (slow, large)
    Truck / Lorry           : 0.18  (slowest)

  Formula:
    clearance_time = (cars/0.38) + (bikes/0.50) + (buses/0.22) + (trucks/0.18)
    green_time     = clearance_time + startup_delay (3s) + safety_buffer (2s)
    green_time     = capped at MAX 100 seconds (AP norm)

  Waiting time per vehicle:
    First vehicle  = startup_delay only (3s)
    Vehicle N      = (N-1) * avg_headway_time
    avg_headway    = green_time / total_vehicles

USAGE:
  # From an image file:
  python src/wait_time_estimator.py --image "C:\\path\\to\\traffic.jpg"

  # From a video file:
  python src/wait_time_estimator.py --video "C:\\path\\to\\traffic.mp4"

  # From webcam:
  python src/wait_time_estimator.py --webcam

  # From video - headless (no window, just print results):
  python src/wait_time_estimator.py --video "traffic.mp4" --no-window

INSTALL:
  pip install ultralytics opencv-python numpy
"""

import cv2
import argparse
import sys
import time
import os
import numpy as np
from datetime import datetime

# ── Try importing YOLO ────────────────────────────────────────────────────
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("[ERROR] ultralytics not installed. Run: pip install ultralytics")

# ── CONSTANTS ─────────────────────────────────────────────────────────────

# Saturation flow rate = vehicles cleared per second at junction
# Based on IRC:93 (Indian Roads Congress) standards adapted for AP roads
SATURATION_FLOW = {
    "bike":  0.50,   # 2-wheelers + autos clear fastest
    "car":   0.38,   # standard car
    "bus":   0.22,   # APSRTC bus - slow acceleration, large size
    "truck": 0.18,   # lorry / goods vehicle - slowest
}

STARTUP_DELAY   = 3.0   # seconds lost at start of green (vehicles start moving)
SAFETY_BUFFER   = 2.0   # seconds added at end for clearance
MIN_GREEN       = 10    # minimum green time (seconds)
MAX_GREEN       = 100   # maximum green time - AP junction norm (seconds)
CONFIDENCE      = 0.15  # YOLO confidence - low for aerial/top-down footage
MODEL_NAME      = "yolov8s.pt"

# YOLO COCO class IDs
VEHICLE_CLASSES = {
    1: "bike",   # bicycle
    2: "car",    # car
    3: "bike",   # motorbike / 2-wheeler
    5: "bus",    # bus
    7: "truck",  # truck / lorry
}

# ── CORE CALCULATION ──────────────────────────────────────────────────────

def estimate_wait_time(cars, bikes, buses, trucks, emergency=False):
    """
    Main function — given vehicle counts, returns full timing breakdown.

    Returns a dict with:
      green_time        : total green signal duration (seconds)
      clearance_time    : time needed for all vehicles to clear (seconds)
      wait_per_vehicle  : dict mapping position -> wait seconds
      avg_wait          : average wait across all vehicles (seconds)
      max_wait          : worst case wait (last vehicle in queue)
      total_vehicles    : total count
      reason            : human-readable explanation of the decision
      congestion        : CLEAR / LOW / MEDIUM / HIGH / CRITICAL
    """
    total = cars + bikes + buses + trucks

    # ── Emergency: immediate green, near-zero wait ─────────────────────
    if emergency:
        return {
            "green_time":       MAX_GREEN,
            "clearance_time":   MAX_GREEN,
            "avg_wait":         0,
            "max_wait":         0,
            "wait_per_vehicle": {},
            "total_vehicles":   total,
            "reason":           "EMERGENCY - GGH Ambulance / Police: immediate priority",
            "congestion":       get_congestion(total),
            "breakdown": {
                "cars": cars, "bikes": bikes, "buses": buses, "trucks": trucks
            }
        }

    if total == 0:
        return {
            "green_time":       MIN_GREEN,
            "clearance_time":   0,
            "avg_wait":         0,
            "max_wait":         0,
            "wait_per_vehicle": {},
            "total_vehicles":   0,
            "reason":           "Road clear - minimum green",
            "congestion":       "CLEAR",
            "breakdown": {
                "cars": 0, "bikes": 0, "buses": 0, "trucks": 0
            }
        }

    # ── Clearance time: how long to clear ALL vehicles ─────────────────
    # Each vehicle type contributes differently based on saturation flow
    clearance_time = (
        (bikes  / SATURATION_FLOW["bike"])  +
        (cars   / SATURATION_FLOW["car"])   +
        (buses  / SATURATION_FLOW["bus"])   +
        (trucks / SATURATION_FLOW["truck"])
    )

    # ── Green time = clearance + startup delay + safety buffer ─────────
    green_time = clearance_time + STARTUP_DELAY + SAFETY_BUFFER

    # ── Apply vehicle-mix rules ────────────────────────────────────────
    bike_ratio  = bikes / total
    heavy_ratio = (buses + trucks) / total

    if bike_ratio > 0.50:
        # 2-wheeler dominant: they clear fast, reduce green time
        green_time *= 0.52
        reason = (f"2-Wheeler dominant ({round(bike_ratio*100)}% bikes/autos) "
                  f"- fast clearing, reduced wait")
    elif heavy_ratio > 0.35:
        # Heavy vehicle dominant: APSRTC buses / lorries need more time
        green_time *= 1.40
        reason = (f"APSRTC/Lorry heavy ({round(heavy_ratio*100)}% heavy) "
                  f"- extended green for safe clearance")
    else:
        reason = f"Mixed traffic - {total} vehicles proportional timing"

    # ── Hard cap: never exceed 100 seconds ─────────────────────────────
    green_time = round(max(MIN_GREEN, min(MAX_GREEN, green_time)))

    # ── Wait time per vehicle position in queue ────────────────────────
    # Build a queue sorted by vehicle type (bikes first, trucks last)
    queue = (["bike"] * bikes +
             ["car"]  * cars  +
             ["bus"]  * buses +
             ["truck"]* trucks)

    wait_per_vehicle = {}
    headway = green_time / max(total, 1)  # avg seconds between each vehicle departing

    for pos, vtype in enumerate(queue):
        if pos == 0:
            wait_per_vehicle[pos + 1] = round(STARTUP_DELAY, 1)
        else:
            wait_per_vehicle[pos + 1] = round(STARTUP_DELAY + pos * headway, 1)

    avg_wait = round(sum(wait_per_vehicle.values()) / len(wait_per_vehicle), 1) if wait_per_vehicle else 0
    max_wait = max(wait_per_vehicle.values()) if wait_per_vehicle else 0

    return {
        "green_time":       green_time,
        "clearance_time":   round(clearance_time, 1),
        "avg_wait":         avg_wait,
        "max_wait":         round(max_wait, 1),
        "wait_per_vehicle": wait_per_vehicle,
        "total_vehicles":   total,
        "reason":           reason,
        "congestion":       get_congestion(total),
        "breakdown": {
            "cars": cars, "bikes": bikes, "buses": buses, "trucks": trucks
        }
    }

def get_congestion(total):
    if total == 0:  return "CLEAR"
    if total <= 5:  return "LOW"
    if total <= 12: return "MEDIUM"
    if total <= 20: return "HIGH"
    return "CRITICAL"

# ── YOLO DETECTION ────────────────────────────────────────────────────────

def detect_vehicles(model, frame):
    """Run YOLO on a frame, return vehicle counts."""
    results  = model(frame, verbose=False, conf=CONFIDENCE)
    counts   = {"cars": 0, "bikes": 0, "buses": 0, "trucks": 0}
    detected = []   # list of (x1,y1,x2,y2,cls_name,conf)

    for box_data in results[0].boxes:
        cls_id = int(box_data.cls[0])
        conf   = float(box_data.conf[0])
        if cls_id not in VEHICLE_CLASSES:
            continue
        vtype  = VEHICLE_CLASSES[cls_id]
        x1,y1,x2,y2 = map(int, box_data.xyxy[0])
        if vtype == "car":    counts["cars"]   += 1
        elif vtype == "bike": counts["bikes"]  += 1
        elif vtype == "bus":  counts["buses"]  += 1
        elif vtype == "truck":counts["trucks"] += 1
        detected.append((x1, y1, x2, y2, vtype, conf))

    return counts, detected, results

# ── DRAW OVERLAY ──────────────────────────────────────────────────────────

def draw_wait_overlay(frame, counts, result):
    """Draw the wait time estimation panel on the frame."""
    h, w = frame.shape[:2]

    # ── Top panel background ──────────────────────────────────────────
    panel_h = 160
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, panel_h), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

    # ── Congestion color ──────────────────────────────────────────────
    cong_colors = {
        "CLEAR":    (0, 230, 180),
        "LOW":      (0, 220, 80),
        "MEDIUM":   (0, 180, 255),
        "HIGH":     (0, 100, 255),
        "CRITICAL": (0, 50, 220),
    }
    c_color = cong_colors.get(result["congestion"], (255, 255, 255))

    # ── Title ─────────────────────────────────────────────────────────
    cv2.putText(frame, "WAIT TIME ESTIMATOR - Smart Traffic AP",
                (10, 22), cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 255, 255), 1)

    # ── Vehicle counts row ────────────────────────────────────────────
    cv2.putText(frame,
                f"Cars: {counts['cars']}   Bikes: {counts['bikes']}   "
                f"Buses: {counts['buses']}   Trucks: {counts['trucks']}   "
                f"Total: {result['total_vehicles']}",
                (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 220, 255), 1)

    # ── Congestion level ──────────────────────────────────────────────
    cv2.putText(frame,
                f"Congestion: {result['congestion']}",
                (10, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.6, c_color, 2)

    # ── Green time ────────────────────────────────────────────────────
    cv2.putText(frame,
                f"Green Signal Duration: {result['green_time']}s",
                (10, 98), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 100), 2)

    # ── Wait times ────────────────────────────────────────────────────
    cv2.putText(frame,
                f"Avg Wait: {result['avg_wait']}s   "
                f"Max Wait (last vehicle): {result['max_wait']}s   "
                f"Clearance: {result['clearance_time']}s",
                (10, 124), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 200, 100), 1)

    # ── Reason ────────────────────────────────────────────────────────
    cv2.putText(frame,
                f"Decision: {result['reason'][:75]}",
                (10, 148), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 180, 180), 1)

    # ── Wait time bar at bottom ───────────────────────────────────────
    bar_y  = h - 40
    bar_h  = 28
    cv2.rectangle(frame, (0, bar_y - 5), (w, h), (15, 15, 15), -1)

    if result["green_time"] > 0:
        # Green bar showing how much of the 100s max is used
        bar_w = int((result["green_time"] / MAX_GREEN) * (w - 20))
        cv2.rectangle(frame, (10, bar_y), (10 + bar_w, bar_y + bar_h), (0, 180, 80), -1)
        cv2.rectangle(frame, (10, bar_y), (w - 10, bar_y + bar_h), (60, 60, 60), 1)
        cv2.putText(frame,
                    f"{result['green_time']}s / {MAX_GREEN}s max",
                    (15, bar_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return frame

# ── PRINT RESULTS TO TERMINAL ─────────────────────────────────────────────

def print_results(counts, result, source=""):
    total = result["total_vehicles"]
    print("")
    print("=" * 60)
    print(f"  WAIT TIME ESTIMATION RESULT")
    if source:
        print(f"  Source: {source}")
    print(f"  Time  : {datetime.now().strftime('%H:%M:%S')}")
    print("=" * 60)
    print(f"  Vehicle Counts:")
    print(f"    Bikes/2-Wheelers : {counts['bikes']:3d}")
    print(f"    Cars             : {counts['cars']:3d}")
    print(f"    Buses (APSRTC)   : {counts['buses']:3d}")
    print(f"    Trucks/Lorries   : {counts['trucks']:3d}")
    print(f"    TOTAL            : {total:3d}")
    print("")
    print(f"  Congestion Level   : {result['congestion']}")
    print(f"  Clearance Time     : {result['clearance_time']}s")
    print(f"  Green Signal Time  : {result['green_time']}s (max 100s AP norm)")
    print(f"  Avg Wait Per Vehicle: {result['avg_wait']}s")
    print(f"  Max Wait (last veh) : {result['max_wait']}s")
    print("")
    print(f"  Decision: {result['reason']}")
    print("")

    # Show wait time for first 10 vehicles in queue
    if result["wait_per_vehicle"]:
        print(f"  Wait time by queue position:")
        queue_items = list(result["wait_per_vehicle"].items())
        show_n = min(10, len(queue_items))
        for pos, wait in queue_items[:show_n]:
            bar = "#" * int(wait / 2)
            print(f"    Vehicle #{pos:3d} : {wait:5.1f}s  {bar}")
        if len(queue_items) > 10:
            print(f"    ... ({len(queue_items) - 10} more vehicles)")
    print("=" * 60)
    print("")

# ── IMAGE MODE ────────────────────────────────────────────────────────────

def run_image(image_path, show_window):
    if not YOLO_AVAILABLE:
        print("[ERROR] Run: pip install ultralytics")
        sys.exit(1)

    if not os.path.exists(image_path):
        print(f"[ERROR] Image not found: {image_path}")
        sys.exit(1)

    print(f"\n[INFO] Loading model {MODEL_NAME}...")
    model = YOLO(MODEL_NAME)
    print(f"[OK]   Model loaded.")

    frame = cv2.imread(image_path)
    if frame is None:
        print(f"[ERROR] Could not read image: {image_path}")
        sys.exit(1)

    print(f"[INFO] Detecting vehicles in image...")
    counts, detected, results = detect_vehicles(model, frame)

    # Calculate wait times
    result = estimate_wait_time(
        cars=counts["cars"], bikes=counts["bikes"],
        buses=counts["buses"], trucks=counts["trucks"]
    )

    # Print to terminal
    print_results(counts, result, source=os.path.basename(image_path))

    # Draw on image and show
    if show_window:
        annotated = results[0].plot()
        output    = draw_wait_overlay(annotated, counts, result)

        # Save output image
        out_path = image_path.replace(".", "_wait_estimate.")
        cv2.imwrite(out_path, output)
        print(f"[OK]   Saved output image: {out_path}")

        cv2.imshow("Wait Time Estimation - Smart Traffic AP", output)
        print("[INFO] Press any key to close the window.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# ── VIDEO MODE ────────────────────────────────────────────────────────────

def run_video(video_source, show_window):
    if not YOLO_AVAILABLE:
        print("[ERROR] Run: pip install ultralytics")
        sys.exit(1)

    print(f"\n[INFO] Loading model {MODEL_NAME}...")
    model = YOLO(MODEL_NAME)
    print(f"[OK]   Model loaded.")

    # Open video
    if video_source == "webcam":
        cap    = cv2.VideoCapture(0)
        source = "Webcam"
    else:
        if not os.path.exists(str(video_source)):
            print(f"[ERROR] Video not found: {video_source}")
            sys.exit(1)
        cap    = cv2.VideoCapture(video_source)
        source = os.path.basename(str(video_source))

    if not cap.isOpened():
        print(f"[ERROR] Could not open: {video_source}")
        sys.exit(1)

    fps    = cap.get(cv2.CAP_PROP_FPS) or 25
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[OK]   Video: {width}x{height} @ {fps:.0f}fps")
    print(f"[INFO] Press Q to stop.\n")

    frame_count  = 0
    FRAME_SKIP   = 3       # process every 3rd frame
    last_print   = time.time()
    PRINT_INTERVAL = 2.0   # print results every 2 seconds

    last_counts = {"cars": 0, "bikes": 0, "buses": 0, "trucks": 0}
    last_result = estimate_wait_time(0, 0, 0, 0)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                # Loop video file
                if video_source != "webcam":
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    break

            frame_count += 1

            # Skip frames for CPU performance
            if frame_count % FRAME_SKIP != 0:
                if show_window:
                    display = draw_wait_overlay(frame.copy(), last_counts, last_result)
                    cv2.imshow("Wait Time Estimator - Smart Traffic AP", display)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                continue

            # Detect vehicles
            counts, detected, results = detect_vehicles(model, frame)

            # Calculate wait times
            result = estimate_wait_time(
                cars=counts["cars"], bikes=counts["bikes"],
                buses=counts["buses"], trucks=counts["trucks"]
            )

            last_counts = counts
            last_result = result

            # Print to terminal every 2 seconds
            now = time.time()
            if now - last_print >= PRINT_INTERVAL:
                print_results(counts, result, source=source)
                last_print = now

            # Show annotated window
            if show_window:
                annotated = results[0].plot()
                display   = draw_wait_overlay(annotated, counts, result)
                cv2.imshow("Wait Time Estimator - Smart Traffic AP", display)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    except KeyboardInterrupt:
        print("\n[INFO] Stopped.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[OK]   Estimator stopped.")

# ── QUICK TEST (no camera/video needed) ──────────────────────────────────

def run_quick_test():
    """
    Test the wait time calculation with sample AP traffic scenarios.
    No camera or video required - runs instantly.
    """
    print("")
    print("=" * 60)
    print("  WAIT TIME ESTIMATOR - Quick Test")
    print("  Sample AP Road Scenarios")
    print("=" * 60)

    scenarios = [
        {
            "name": "Brodipet - Morning Rush (2-Wheeler Dominant)",
            "cars": 4, "bikes": 14, "buses": 2, "trucks": 1, "emergency": False
        },
        {
            "name": "Benz Circle - Afternoon (Mixed Heavy Traffic)",
            "cars": 6, "bikes": 5,  "buses": 5, "trucks": 3, "emergency": False
        },
        {
            "name": "NH-16 - Evening (Lorry/Bus Dominant)",
            "cars": 3, "bikes": 2,  "buses": 6, "trucks": 5, "emergency": False
        },
        {
            "name": "Lakshmipuram - Market Hours (Auto Dominant)",
            "cars": 3, "bikes": 11, "buses": 1, "trucks": 1, "emergency": False
        },
        {
            "name": "ANY LANE - Emergency (GGH Ambulance)",
            "cars": 5, "bikes": 4,  "buses": 2, "trucks": 1, "emergency": True
        },
        {
            "name": "Late Night - Light Traffic",
            "cars": 2, "bikes": 3,  "buses": 0, "trucks": 1, "emergency": False
        },
    ]

    for s in scenarios:
        result = estimate_wait_time(
            cars=s["cars"], bikes=s["bikes"],
            buses=s["buses"], trucks=s["trucks"],
            emergency=s["emergency"]
        )
        total = s["cars"] + s["bikes"] + s["buses"] + s["trucks"]
        print(f"\n  Scenario: {s['name']}")
        print(f"  Vehicles : Cars={s['cars']} Bikes={s['bikes']} Buses={s['buses']} Trucks={s['trucks']} (Total={total})")
        print(f"  Congestion    : {result['congestion']}")
        print(f"  Green Time    : {result['green_time']}s")
        print(f"  Clearance     : {result['clearance_time']}s")
        print(f"  Avg Wait      : {result['avg_wait']}s per vehicle")
        print(f"  Max Wait      : {result['max_wait']}s (last vehicle)")
        print(f"  Decision      : {result['reason']}")
        print(f"  {'-'*56}")

    print("")

# ── ENTRY POINT ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Wait Time Estimator - Smart Traffic AP (Guntur/Vijayawada)",
        formatter_class=argparse.RawTextHelpFormatter
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--image", type=str, default=None,
        help=(
            "Path to a traffic image file.\n"
            "Example: C:\\Users\\Name\\Desktop\\traffic.jpg"
        )
    )
    group.add_argument(
        "--video", type=str, default=None,
        help=(
            "Path to a traffic video file.\n"
            "Example: C:\\Users\\Name\\Desktop\\traffic.mp4"
        )
    )
    group.add_argument(
        "--webcam", action="store_true",
        help="Use laptop webcam as video source."
    )
    group.add_argument(
        "--test", action="store_true",
        help="Run quick test with sample AP traffic scenarios (no camera needed)."
    )
    parser.add_argument(
        "--no-window", action="store_true",
        help="Run without display window - print results to terminal only."
    )
    parser.add_argument(
        "--conf", type=float, default=CONFIDENCE,
        help=f"YOLO confidence threshold (default: {CONFIDENCE})"
    )
    parser.add_argument(
        "--model", type=str, default=MODEL_NAME,
        choices=["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"],
        help="YOLO model: n=fastest, s=balanced, m=most accurate (default: yolov8s.pt)"
    )

    args = parser.parse_args()

    # Override globals
    CONFIDENCE = args.conf
    MODEL_NAME = args.model

    if args.test or (not args.image and not args.video and not args.webcam):
        # Default: run quick test if no input given
        run_quick_test()

    elif args.image:
        run_image(args.image, show_window=not args.no_window)

    elif args.video:
        run_video(args.video, show_window=not args.no_window)

    elif args.webcam:
        run_video("webcam", show_window=not args.no_window)
