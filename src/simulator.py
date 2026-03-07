"""
simulator.py
────────────
Fallback simulator when no real camera/stream is available.
Generates realistic AP traffic data for 4 lanes based on
Guntur/Vijayawada road profiles.

Used by app.py automatically if no real detector is running.
"""

import math, random, sqlite3, threading, time
from datetime import datetime
from src.signal_logic import schedule_junction, get_congestion_level

AP_PROFILES = {
    1: {"name":"Brodipet",       "direction":"↑ Amaravati Rd", "car":.22,"bike":.55,"bus":.13,"truck":.10,"density":16},
    2: {"name":"Benz Circle",    "direction":"→ MG Road",      "car":.28,"bike":.35,"bus":.25,"truck":.12,"density":18},
    3: {"name":"NH-16 Vjwada",   "direction":"↓ Vijayawada",   "car":.25,"bike":.20,"bus":.28,"truck":.27,"density":14},
    4: {"name":"Lakshmipuram",   "direction":"← Besant Rd",    "car":.20,"bike":.58,"bus":.12,"truck":.10,"density":12},
}

def gauss():
    u,v=0,0
    while not u: u=random.random()
    while not v: v=random.random()
    return math.sqrt(-2*math.log(u))*math.cos(2*math.pi*v)

class TrafficSimulator:
    def __init__(self, db_path="traffic.db"):
        self.db_path = db_path
        self.tick = 0
        self._lock = threading.Lock()
        self._running = False
        self.latest = []
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS vehicle_count (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                lane INTEGER, lane_name TEXT, timestamp TEXT,
                cars INTEGER, bikes INTEGER, buses INTEGER, trucks INTEGER,
                total INTEGER, emergency INTEGER DEFAULT 0,
                congestion_level TEXT, signal_state TEXT DEFAULT 'PENDING',
                green_duration INTEGER DEFAULT 0, wait_time INTEGER DEFAULT 0,
                reason TEXT DEFAULT '', priority_score INTEGER DEFAULT 0
            )""")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS signal_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT, lane INTEGER, lane_name TEXT,
                green_duration INTEGER, reason TEXT
            )""")
        conn.commit(); conn.close()

    def _gen_lane(self, lid):
        p = AP_PROFILES[lid]
        phase = self.tick*0.07 + lid*1.3
        density = p["density"]*(0.55+0.45*abs(math.sin(phase)))
        t = max(0, density+gauss()*2)
        return {
            "lane_id": lid, "lane_name": p["name"], "direction": p["direction"],
            "cars":   max(0,round(t*p["car"]  +gauss()*.6)),
            "bikes":  max(0,round(t*p["bike"] +gauss()*.8)),
            "buses":  max(0,round(t*p["bus"]  +gauss()*.4)),
            "trucks": max(0,round(t*p["truck"]+gauss()*.3)),
            "emergency": random.random() < 0.038,
        }

    def tick_once(self):
        lanes = [self._gen_lane(i) for i in range(1,5)]
        scheduled = schedule_junction(lanes)
        conn = sqlite3.connect(self.db_path)
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for l in scheduled:
            total = l["cars"]+l["bikes"]+l["buses"]+l["trucks"]
            conn.execute("""
                INSERT INTO vehicle_count
                (lane,lane_name,timestamp,cars,bikes,buses,trucks,total,
                 emergency,congestion_level,signal_state,green_duration,
                 wait_time,reason,priority_score)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (l["lane_id"],l["lane_name"],ts,l["cars"],l["bikes"],
                 l["buses"],l["trucks"],total,1 if l["emergency"] else 0,
                 l["congestion_level"],l["signal_state"],l["green_duration"],
                 l["wait_time"],l["reason"],l["priority_score"]))
            if l["signal_state"]=="GREEN":
                conn.execute("""
                    INSERT INTO signal_log(timestamp,lane,lane_name,green_duration,reason)
                    VALUES(?,?,?,?,?)""",
                    (ts,l["lane_id"],l["lane_name"],l["green_duration"],l["reason"]))
        conn.commit(); conn.close()
        self.tick += 1
        with self._lock: self.latest = scheduled
        return scheduled

    def get_snapshot(self):
        with self._lock: return list(self.latest)

    def start_background(self, interval=2.5):
        self._running = True
        def _loop():
            while self._running:
                self.tick_once(); time.sleep(interval)
        threading.Thread(target=_loop, daemon=True).start()
        print("✅ Simulator running (fallback mode — no real stream)")

    def stop(self): self._running = False
