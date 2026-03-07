"""
signal_logic.py
───────────────
Core signal timing intelligence for Guntur / Vijayawada junctions.

Rules (in priority order):
  1. Emergency vehicle (ambulance / police) → immediate 100s green
  2. 2-wheeler / auto dominant (>50%)       → reduced green (they clear fast)
  3. APSRTC bus / lorry dominant (>35%)     → extended green (they need time)
  4. Mixed traffic                          → proportional to vehicle count

Hard cap: MAX 100 seconds — AP junction norm.
"""

MIN_GREEN       = 10
MAX_GREEN       = 100   # AP traffic norm — never exceed 100s
EMERGENCY_GREEN = 100   # emergency also capped at 100s

def get_congestion_level(total: int) -> str:
    if total == 0:    return "CLEAR"
    if total <= 5:    return "LOW"
    if total <= 12:   return "MEDIUM"
    if total <= 20:   return "HIGH"
    return "CRITICAL"

def compute_green_duration(lane: dict) -> tuple:
    """
    Returns (green_seconds: int, reason: str)
    lane dict keys: cars, bikes, buses, trucks, emergency (bool)
    """
    cars      = lane.get("cars", 0)
    bikes     = lane.get("bikes", 0)
    buses     = lane.get("buses", 0)
    trucks    = lane.get("trucks", 0)
    emergency = lane.get("emergency", False)
    total     = cars + bikes + buses + trucks

    # Rule 1 — Emergency override
    if emergency:
        return (EMERGENCY_GREEN,
                "🚨 GGH Ambulance / Police — immediate priority")

    if total == 0:
        return (MIN_GREEN, "Road clear — minimum green")

    # Base proportional time
    dur = 15 + total * 2.8

    bike_ratio  = bikes / total
    heavy_ratio = (buses + trucks) / total

    # Rule 2 — 2-wheeler / auto dominant
    if bike_ratio > 0.50:
        dur *= 0.52
        reason = (f"🛵 2-Wheeler dominant ({round(bike_ratio*100)}%)"
                  f" — fast clearing, reduced wait")

    # Rule 3 — APSRTC buses / lorries dominant
    elif heavy_ratio > 0.35:
        dur *= 1.40
        reason = (f"🚌 APSRTC/Lorry heavy ({round(heavy_ratio*100)}%)"
                  f" — extended green")

    else:
        reason = f"🚗 Mixed traffic — {total} vehicles"

    dur = int(max(MIN_GREEN, min(MAX_GREEN, round(dur))))
    return (dur, reason)


def schedule_junction(lanes: list) -> list:
    """
    Input : list of lane dicts with vehicle counts + emergency flag
    Output: same list with signal_state, green_duration, wait_time, reason added
    """
    scored = []
    for lane in lanes:
        cars      = lane.get("cars", 0)
        bikes     = lane.get("bikes", 0)
        buses     = lane.get("buses", 0)
        trucks    = lane.get("trucks", 0)
        emergency = lane.get("emergency", False)

        # Priority score — emergency always first
        score = 99999 if emergency else \
                (cars * 3) + (bikes * 1) + (buses * 7) + (trucks * 6)

        dur, reason = compute_green_duration(lane)
        scored.append({**lane,
                       "priority_score": score,
                       "green_duration": dur,
                       "reason": reason})

    scored.sort(key=lambda x: x["priority_score"], reverse=True)

    result, elapsed = [], 0
    for i, lane in enumerate(scored):
        total = lane.get("cars",0)+lane.get("bikes",0)+lane.get("buses",0)+lane.get("trucks",0)
        result.append({**lane,
                       "queue_position":  i + 1,
                       "signal_state":    "GREEN" if i == 0 else "RED",
                       "wait_time":       elapsed,
                       "congestion_level": get_congestion_level(total)})
        elapsed += lane["green_duration"]

    return result
