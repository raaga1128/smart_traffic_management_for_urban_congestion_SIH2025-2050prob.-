"""
app.py — Smart Traffic Flask Server
─────────────────────────────────────
Serves the real-time dashboard and REST API.

Data priority:
  1. If youtube_detector.py is running → reads REAL vehicle counts from DB
  2. If no detector running → starts simulator as fallback automatically

Run:
  python app.py
  → Open http://localhost:5000
"""

import os, sys, json, sqlite3, time, threading
from datetime import datetime
from flask import Flask, render_template, jsonify, Response

sys.path.insert(0, os.path.dirname(__file__))
from src.signal_logic import schedule_junction, get_congestion_level
from src.simulator import TrafficSimulator

app  = Flask(__name__, template_folder="templates")
DB   = "traffic.db"

# ── Start simulator as fallback ───────────────────────────────────────────
# If real detector (youtube_detector.py) is writing to the DB, the simulator
# still runs but its data gets mixed in. For pure real-data mode, pass
# USE_SIMULATOR=false when running: USE_SIMULATOR=false python app.py
USE_SIM = os.environ.get("USE_SIMULATOR", "true").lower() != "false"
sim = None
if USE_SIM:
    sim = TrafficSimulator(db_path=DB)
    sim.start_background(interval=2.5)

def db():
    c = sqlite3.connect(DB); c.row_factory = sqlite3.Row; return c

# ── Helper: check if real detector is writing fresh data ─────────────────
def real_data_active():
    """Returns True if youtube_detector.py wrote data in the last 5 seconds."""
    try:
        conn = db()
        row = conn.execute(
            "SELECT timestamp FROM vehicle_count ORDER BY id DESC LIMIT 1"
        ).fetchone()
        conn.close()
        if not row: return False
        ts = datetime.strptime(row["timestamp"], "%Y-%m-%d %H:%M:%S")
        age = (datetime.now() - ts).total_seconds()
        return age < 5
    except: return False

# ── Routes ────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("dashboard.html")

@app.route("/api/snapshot")
def api_snapshot():
    """
    Latest state for all 4 lanes.
    If real detector is active → read last row per lane from DB and run signal logic.
    If only simulator → use simulator snapshot.
    """
    if real_data_active():
        # Read latest counts per lane from DB, then run signal logic
        conn = db()
        lanes = []
        for lane_id in range(1, 5):
            row = conn.execute("""
                SELECT * FROM vehicle_count WHERE lane=?
                ORDER BY id DESC LIMIT 1
            """, (lane_id,)).fetchone()
            if row:
                lanes.append({
                    "lane_id":   row["lane"],
                    "lane_name": row["lane_name"] or f"Lane {lane_id}",
                    "cars":      row["cars"],
                    "bikes":     row["bikes"],
                    "buses":     row["buses"],
                    "trucks":    row["trucks"],
                    "emergency": bool(row["emergency"]),
                })
        conn.close()
        if lanes:
            scheduled = schedule_junction(lanes)
            out = []
            for l in scheduled:
                total = l["cars"]+l["bikes"]+l["buses"]+l["trucks"]
                out.append({
                    "lane_id":        l["lane_id"],
                    "lane_name":      l["lane_name"],
                    "cars":           l["cars"],
                    "bikes":          l["bikes"],
                    "buses":          l["buses"],
                    "trucks":         l["trucks"],
                    "total":          total,
                    "emergency":      l["emergency"],
                    "congestion":     l["congestion_level"],
                    "signal":         l["signal_state"],
                    "green_duration": l["green_duration"],
                    "wait_time":      l["wait_time"],
                    "reason":         l["reason"],
                    "priority":       l["priority_score"],
                    "data_source":    "REAL",   # tells dashboard this is live
                })
            return jsonify(out)

    # Fallback: simulator data
    if sim:
        snapshot = sim.get_snapshot()
        out = []
        for l in snapshot:
            total = l["cars"]+l["bikes"]+l["buses"]+l["trucks"]
            out.append({
                "lane_id":        l["lane_id"],
                "lane_name":      l["lane_name"],
                "cars":           l["cars"],
                "bikes":          l["bikes"],
                "buses":          l["buses"],
                "trucks":         l["trucks"],
                "total":          total,
                "emergency":      l["emergency"],
                "congestion":     l["congestion_level"],
                "signal":         l["signal_state"],
                "green_duration": l["green_duration"],
                "wait_time":      l["wait_time"],
                "reason":         l["reason"],
                "priority":       l["priority_score"],
                "data_source":    "SIM",   # tells dashboard this is simulated
            })
        return jsonify(out)

    return jsonify([])

@app.route("/api/history")
def api_history():
    conn = db()
    rows = conn.execute("""
        SELECT lane, lane_name, timestamp, total, congestion_level
        FROM vehicle_count ORDER BY id DESC LIMIT 240
    """).fetchall()
    conn.close()
    data = {}
    for r in rows:
        lid = r["lane"]
        if lid not in data:
            data[lid] = {"name": r["lane_name"], "points": []}
        data[lid]["points"].append({
            "t": r["timestamp"], "total": r["total"],
            "congestion": r["congestion_level"],
        })
    for lid in data:
        data[lid]["points"] = data[lid]["points"][::-1]
    return jsonify(data)

@app.route("/api/stats")
def api_stats():
    conn = db()
    today = datetime.now().strftime("%Y-%m-%d")
    def q(sql, *args):
        r = conn.execute(sql, args).fetchone()
        return r[0] if r and r[0] is not None else 0

    total_today   = q("SELECT SUM(total)    FROM vehicle_count WHERE timestamp LIKE ?", f"{today}%")
    emergency_cnt = q("SELECT COUNT(*)      FROM vehicle_count WHERE emergency=1 AND timestamp LIKE ?", f"{today}%")
    avg_green     = q("SELECT AVG(green_duration) FROM vehicle_count WHERE timestamp LIKE ? AND green_duration > 0", f"{today}%")
    signal_cnt    = q("SELECT COUNT(*)      FROM signal_log WHERE timestamp LIKE ?", f"{today}%")
    conn.close()

    is_live = real_data_active()
    return jsonify({
        "total_vehicles":   int(total_today),
        "emergency_count":  int(emergency_cnt),
        "avg_green":        round(avg_green, 1),
        "signal_decisions": int(signal_cnt),
        "data_mode":        "LIVE" if is_live else "SIMULATION",
    })

@app.route("/api/signal_log")
def api_signal_log():
    conn = db()
    rows = conn.execute("""
        SELECT timestamp, lane, lane_name, green_duration, reason
        FROM signal_log ORDER BY id DESC LIMIT 20
    """).fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])

@app.route("/api/status")
def api_status():
    """Simple health check — tells dashboard if real data or simulator is active."""
    return jsonify({
        "real_data": real_data_active(),
        "simulator": sim is not None,
        "time": datetime.now().strftime("%H:%M:%S"),
    })

if __name__ == "__main__":
    print("\n" + "═"*55)
    print("  🚦  Smart Traffic — Guntur / Vijayawada")
    print("  Dashboard  → http://localhost:5000")
    print("  Data mode  →", "Simulator (start youtube_detector.py for real data)" if USE_SIM else "Real data only")
    print("═"*55 + "\n")
    app.run(debug=False, threaded=True, port=5000)
