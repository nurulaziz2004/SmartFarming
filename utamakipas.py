# ====== Install ======
# pip install paho-mqtt==2.1.0 flask==3.1.1 scikit-learn==1.5.2 matplotlib==3.9.2 pandas==2.2.3

import os
import time
import threading
from datetime import datetime



try :
    import paho.mqtt.client as mqtt
    from flask import Flask, render_template_string, request, jsonify
    import pandas as pd
    import numpy as np
    from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use('Agg')  # gunakan backend non-GUI
except :
    print("ADA LIBRARY GAGAL, MULAI MENGINSTAL ..... ")
    time.sleep(5)
    os.system("pip install paho-mqtt==2.1.0 flask==3.1.1 scikit-learn==1.5.2 matplotlib==3.9.2 pandas==2.2.3")
    import paho.mqtt.client as mqtt
    from flask import Flask, render_template_string, request, jsonify
    import pandas as pd
    import numpy as np
    from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use('Agg')  # gunakan backend non-GUI




# ===================== Konfigurasi =====================
BROKER = "test.mosquitto.org"
PORT = 1883
TOPIC_BASE = "SatriaSensors773546"
current_folder = os.path.dirname(os.path.abspath(__file__))
print("current FOLDER : ",current_folder)


# Topik relay
RELAY_TOPICS = {
    1: f"{TOPIC_BASE}/relay1",
    2: f"{TOPIC_BASE}/relay2",
    3: f"{TOPIC_BASE}/relay3",
    4: f"{TOPIC_BASE}/relay4",
}

# Topik sensor
sensor_topics = [
    f"{TOPIC_BASE}/ldr",
    f"{TOPIC_BASE}/suhu",
    f"{TOPIC_BASE}/kelembaban",
    f"{TOPIC_BASE}/kelembaban_tanah_1",
    f"{TOPIC_BASE}/kelembaban_tanah_2",
    f"{TOPIC_BASE}/kelembaban_tanah_3",
]

SENSOR_KEYS = [t.split("/")[-1] for t in sensor_topics]

# ===================== State Global =====================
sensor_data = {k: None for k in SENSOR_KEYS}
sensor_timestamp = {k: None for k in SENSOR_KEYS}
control_mode = {"mode": "manual"}
relay_state = {1: "OFF", 2: "OFF", 3: "OFF", 4: "OFF"}
decision_info = {
    "ready": False,
    "features": {},
    "prediction": {},
    "proba": {},
    "path_rules": {},
    "updated_at": None,
}

# ===================== MQTT =====================
def on_connect(client, userdata, flags, rc):
    print("Connected with result code", rc)
    for t in sensor_topics:
        client.subscribe(t)
    for t in RELAY_TOPICS.values():
        client.subscribe(t)

def safe_float(s):
    try:
        return float(str(s).strip())
    except:
        return None

def on_message(client, userdata, msg):
    topic = msg.topic
    payload = msg.payload.decode(errors="ignore")
    key = topic.split("/")[-1]

    if key in sensor_data:
        val = safe_float(payload)
        sensor_data[key] = val
        sensor_timestamp[key] = datetime.now().strftime("%H:%M:%S")
    for rid, rtopic in RELAY_TOPICS.items():
        if topic == rtopic:
            print(f"[RELAY-ECHO] relay{rid} <- {payload}")

client = mqtt.Client(client_id="SatriaSensors_FlaskDT", protocol=mqtt.MQTTv311)
client.on_connect = on_connect
client.on_message = on_message
client.connect(BROKER, PORT, 60)
client.loop_start()

def publish_relay(relay_id: int, state: str):
    state = "ON" if str(state).upper() == "ON" else "OFF"
    topic = RELAY_TOPICS.get(relay_id)
    if topic:
        client.publish(topic, state)
        relay_state[relay_id] = state
        print(f"[RELAY] relay{relay_id} => {state}")

# ===================== Decision Tree =====================


df = pd.read_csv(os.path.join(current_folder,"D:\skripsi\Kontrol Kipas\selada sudah ada kipas.csv"))

FEATURES = ["suhu", "kelembaban", "kelembaban_tanah", "intensitas_cahaya"]
TARGET_SIRAM = "label"
TARGET_KIPAS = "kipas_exhaust"   # === NEW ===

X = df[FEATURES]
y_siram = df[TARGET_SIRAM]
y_kipas = df[TARGET_KIPAS]

X_train, X_test, y_train_siram, y_test_siram = train_test_split(X, y_siram, test_size=0.30, random_state=42)
X_train, X_test, y_train_kipas, y_test_kipas = train_test_split(X, y_kipas, test_size=0.30, random_state=42)

# Model untuk penyiraman
model_siram = DecisionTreeClassifier(max_depth=4, random_state=42)
model_siram.fit(X_train, y_train_siram)

# Model untuk kipas/exhaust
model_kipas = DecisionTreeClassifier(max_depth=4, random_state=42)
model_kipas.fit(X_train, y_train_kipas)

print("Akurasi SIRAM:", accuracy_score(y_test_siram, model_siram.predict(X_test)))
print("Akurasi KIPAS:", accuracy_score(y_test_kipas, model_kipas.predict(X_test)))

# Simpan gambar tree penyiraman
os.makedirs(os.path.join(current_folder,"static"), exist_ok=True)
plt.figure(figsize=(13, 7))
plot_tree(model_siram, feature_names=FEATURES, class_names=["Tidak Siram","Siram"], filled=True, rounded=True)
plt.tight_layout()
plt.savefig(os.path.join(current_folder,"static/tree_siram.png"), dpi=140)
plt.close()

# Simpan gambar tree kipas
plt.figure(figsize=(13, 7))
plot_tree(model_kipas, feature_names=FEATURES, class_names=["Mati","Nyala"], filled=True, rounded=True)
plt.tight_layout()
plt.savefig(os.path.join(current_folder,"static/tree_kipas.png"), dpi=140)
plt.close()

tree_rules_siram = export_text(model_siram, feature_names=FEATURES)
tree_rules_kipas = export_text(model_kipas, feature_names=FEATURES)


# ===================== Decision Path =====================
def decision_path_rules(clf: DecisionTreeClassifier, xrow: np.ndarray, features: list):
    tree = clf.tree_
    feature = tree.feature
    threshold = tree.threshold
    node_indicator = clf.decision_path(xrow.reshape(1, -1))
    node_index = node_indicator.indices
    rules = []
    for node_id in node_index:
        if feature[node_id] != -2:
            fname = features[feature[node_id]]
            thr = threshold[node_id]
            if xrow[feature[node_id]] <= thr:
                rules.append(f"{fname} <= {thr:.2f}")
            else:
                rules.append(f"{fname} > {thr:.2f}")
    return rules

# ===================== Loop Otomatis =====================
def collect_latest_features():
    suhu = sensor_data.get("suhu")
    kelembaban = sensor_data.get("kelembaban")
    ldr = sensor_data.get("ldr")
    pot_features = {}
    for i in range(1,4):
        tanah = sensor_data.get(f"kelembaban_tanah_{i}")
        pot_features[f"pot_{i}"] = {
            "suhu": suhu,
            "kelembaban": kelembaban,
            "kelembaban_tanah": tanah,
            "intensitas_cahaya": ldr
        }
    if any(any(v is None for v in pot_features[p].values()) for p in pot_features):
        return None
    return pot_features

def auto_control_loop():
    while True:
        try:
            if control_mode["mode"] == "auto":
                feats = collect_latest_features()
                if feats is not None:
                    kipas_pred_global = 0   # === NEW ===
                    for i in range(1,4):
                        x = np.array([feats[f"pot_{i}"][f] for f in FEATURES], dtype=float)

                        # Prediksi siram
                        pred_siram = int(model_siram.predict([x])[0])
                        proba_siram = model_siram.predict_proba([x])[0].tolist()
                        rules_siram = decision_path_rules(model_siram, x, FEATURES)

                        # Prediksi kipas
                        pred_kipas = int(model_kipas.predict([x])[0])
                        proba_kipas = model_kipas.predict_proba([x])[0].tolist()
                        rules_kipas = decision_path_rules(model_kipas, x, FEATURES)

                        decision_info["features"][f"pot_{i}"] = feats[f"pot_{i}"]
                        decision_info["prediction"][f"pot_{i}"] = {"siram": pred_siram, "kipas": pred_kipas}   # === NEW ===
                        decision_info["proba"][f"pot_{i}"] = {"siram": proba_siram, "kipas": proba_kipas}     # === NEW ===
                        decision_info["path_rules"][f"pot_{i}"] = {"siram": rules_siram, "kipas": rules_kipas} # === NEW ===

                        # Kontrol relay masing-masing pot (pompa)
                        publish_relay(i, "ON" if pred_siram==1 else "OFF")

                        # Kalau ada salah satu pot butuh kipas, nyalakan kipas global
                        if pred_kipas == 1:
                            kipas_pred_global = 1

                    # Relay 4 khusus kipas
                    publish_relay(4, "ON" if kipas_pred_global==1 else "OFF")

                    decision_info["ready"] = True
                    decision_info["updated_at"] = datetime.now().strftime("%H:%M:%S")
            time.sleep(0.5)
        except Exception as e:
            print("[AUTO LOOP ERROR]", e)
            time.sleep(1)

threading.Thread(target=auto_control_loop, daemon=True).start()

# ===================== Flask =====================
app = Flask(__name__)

PAGE = """
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>🌱 Smart Farming Dashboard</title>
<style>
body { font-family: Arial; margin:18px; }
.row { display:flex; flex-wrap: wrap; gap:12px; }
.card { border:1px solid #ddd; border-radius:12px; padding:12px 14px; box-shadow:0 2px 6px rgba(0,0,0,0.05); }
.sensor { width:220px; }
.btn { padding:8px 12px; border:1px solid #333; border-radius:8px; background:#f7f7f7; cursor:pointer; margin-right:6px; }
.btn.active { background:#e8ffe8; border-color:#2b8a3e; }
.muted { color:#777; font-size:12px; }
pre { background:#f5f5f5; padding:10px; border-radius:8px; overflow:auto; }
.tag { display:inline-block; padding:2px 6px; border-radius:6px; background:#eef; margin-right:6px; font-size:12px; }
</style>
<script>
async function setMode(mode){
  await fetch('/mode/'+mode, {method:'POST'});
  refreshAll();
}
async function relay(rid,state){
  await fetch(`/relay/${rid}/${state}`, {method:'POST'});
  refreshAll();
}

async function refreshSensors() {
  let r = await fetch('/sensors');
  let d = await r.json();
  let html = "";
  Object.keys(d.values).forEach(k => {
    let val = d.values[k] ?? '-';
    let unit = (k.toLowerCase().includes("suhu")) ? "°C" : "%";  
    html += `<div class="card sensor"><b>${k}</b><br>${val}${unit}<br></div>`;
  });

  document.getElementById('sensors').innerHTML = html;
  document.getElementById('mode').innerText = d.mode.toUpperCase();
  document.getElementById('btn_manual').classList.toggle('active', d.mode=='manual');
  document.getElementById('btn_auto').classList.toggle('active', d.mode=='auto');
  ['1','2','3','4'].forEach(rid=>{
    let state=d.relay[rid];
    document.getElementById('r'+rid).innerText=state;
    document.getElementById('r'+rid+'_on').classList.toggle('active', state=='ON');
    document.getElementById('r'+rid+'_off').classList.toggle('active', state=='OFF');
  });
}
async function refreshDecision(){
  let r = await fetch('/decision'); let d = await r.json();
  let info = document.getElementById('decision');
  if(!d.ready){ info.innerHTML='<i>Menunggu data lengkap...</i>'; return; }
  let html="";
  for(let i=1;i<=3;i++){
    let p='pot_'+i;
    let feats = d.features[p];
    let pred = d.prediction[p];
    let proba = d.proba[p];
    let rules = d.path_rules[p];
    html+=`<h4>Pot ${i}</h4>
      <div><b>Prediksi Siram:</b> ${pred.siram==1?'SIRAM':'TIDAK SIRAM'}<br>
           <b>Prediksi Kipas:</b> ${pred.kipas==1?'NYALA':'MATI'}</div>
      <div>P(Siram=0)=${proba.siram[0].toFixed(2)}, P(Siram=1)=${proba.siram[1].toFixed(2)}</div>
      <div>P(Kipas=0)=${proba.kipas[0].toFixed(2)}, P(Kipas=1)=${proba.kipas[1].toFixed(2)}</div>
      <div style="margin-top:6px">${Object.entries(feats).map(([k,v])=>`<span class="tag">${k}:${v.toFixed(2)}</span>`).join(' ')}</div>
      <div class="muted" style="margin-top:4px">Updated: ${d.updated_at||''}</div>
      <hr>
      <div><b>Jalur Keputusan Siram:</b><br>${(rules.siram||[]).map(r=>"• "+r).join('<br>')}</div>
      <div><b>Jalur Keputusan Kipas:</b><br>${(rules.kipas||[]).map(r=>"• "+r).join('<br>')}</div>`;
  }
  info.innerHTML=html;
}
async function refreshRules(){
  let r1 = await fetch('/rules_siram'); let t1 = await r1.text();
  let r2 = await fetch('/rules_kipas'); let t2 = await r2.text();
  document.getElementById('rules_siram').innerText=t1;
  document.getElementById('rules_kipas').innerText=t2;
  document.getElementById('tree_siram').src='/static/tree_siram.png?ts='+Date.now();
  document.getElementById('tree_kipas').src='/static/tree_kipas.png?ts='+Date.now();
}
async function refreshAll(){ refreshSensors(); refreshDecision(); refreshRules(); }
setInterval(refreshAll,1000); window.onload=refreshAll;
</script>
</head>
<body>
<h1>🌱 Smart Farming Dashboard</h1>

<div class="card" style="margin-bottom:12px;">
Mode: <b id="mode">-</b>
<button class="btn" id="btn_manual" onclick="setMode('manual')">Manual</button>
<button class="btn" id="btn_auto" onclick="setMode('auto')">Otomatis</button>
<span class="muted">Manual: pakai tombol relay. Otomatis: decision tree.</span>
</div>

<h2>Sensor</h2>
<div id="sensors" class="row"></div>

<h2>Kontrol Relay</h2>
<div class="card">
<div>Relay 1 (Pompa Pot 1): <b id="r1">-</b>
<button class="btn" id="r1_on" onclick="relay(1,'ON')">ON</button>
<button class="btn" id="r1_off" onclick="relay(1,'OFF')">OFF</button></div>

<div>Relay 2 (Pompa Pot 2): <b id="r2">-</b>
<button class="btn" id="r2_on" onclick="relay(2,'ON')">ON</button>
<button class="btn" id="r2_off" onclick="relay(2,'OFF')">OFF</button></div>

<div>Relay 3 (Pompa Pot 3): <b id="r3">-</b>
<button class="btn" id="r3_on" onclick="relay(3,'ON')">ON</button>
<button class="btn" id="r3_off" onclick="relay(3,'OFF')">OFF</button></div>

<div>Relay 4 (Kipas Exhaust): <b id="r4">-</b>
<button class="btn" id="r4_on" onclick="relay(4,'ON')">ON</button>
<button class="btn" id="r4_off" onclick="relay(4,'OFF')">OFF</button></div>

<div class="muted" style="margin-top:6px">Catatan: Mode Otomatis mengatur pompa & kipas sesuai prediksi.</div>
</div>

<h2>Decision Insight (Realtime)</h2>
<div id="decision" class="card"></div>

<h2>Decision Tree Penyiraman</h2>
<img id="tree_siram" src="/static/tree_siram.png" width="900" style="border:1px solid #ddd; border-radius:8px;">
<pre id="rules_siram"></pre>

<h2>Decision Tree Kipas</h2>
<img id="tree_kipas" src="/static/tree_kipas.png" width="900" style="border:1px solid #ddd; border-radius:8px;">
<pre id="rules_kipas"></pre>
</body>
</html>
"""

@app.route("/")
def index(): return render_template_string(PAGE)

@app.route("/mode/<mode>", methods=["POST"])
def set_mode(mode):
    control_mode["mode"] = "auto" if mode.lower().strip()=="auto" else "manual"
    return jsonify(ok=True, mode=control_mode["mode"])

@app.route("/relay/<int:rid>/<state>", methods=["POST"])
def set_relay(rid, state):
    state = "ON" if state.upper() == "ON" else "OFF"
    if rid in relay_state:
        publish_relay(rid, state)   # fungsi publish MQTT
        return jsonify(ok=True, relay=rid, state=relay_state[rid])
    return jsonify(ok=False, error="invalid relay"), 400

@app.route("/sensors")
def sensors_api():
    return jsonify({"values": sensor_data, "timestamps": sensor_timestamp,
                    "mode": control_mode["mode"], "relay": relay_state})

@app.route("/decision")
def decision_api(): return jsonify(decision_info)

@app.route("/rules_siram")
def rules_siram_api(): return tree_rules_siram

@app.route("/rules_kipas")
def rules_kipas_api(): return tree_rules_kipas

if __name__=="__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
