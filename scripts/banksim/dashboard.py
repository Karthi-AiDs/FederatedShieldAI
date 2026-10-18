#!/usr/bin/env python3
import json
from collections import defaultdict
from datetime import datetime, timezone
from html import escape
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs_public"
PORT = 8501

SCENARIOS = {
    "clean_fedavg": ("Clean FedAvg", "Baseline"),
    "attacked_fedavg": ("Attacked FedAvg", "Attack"),
    "defended_fedavg": ("Defended FedAvg", "Defended"),
}


def read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def load_audit():
    path = OUT / "banksim_defense_audit.jsonl"
    records = []
    try:
        with path.open(encoding="utf-8") as f:
            for line in f:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    except OSError:
        pass
    return records


def fmt(value, digits=4):
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "—"


def pct(value):
    try:
        return f"{float(value) * 100:.2f}%"
    except (TypeError, ValueError):
        return "—"


def metric_card(label, value, detail):
    return f"""
    <article class="card">
      <div class="muted">{escape(label)}</div>
      <div class="big">{escape(str(value))}</div>
      <div class="muted small">{escape(detail)}</div>
    </article>"""


def render():
    report = read_json(OUT / "banksim_defense_comparison.json")
    scenarios = report.get("scenarios", {})
    audit = load_audit()

    clean = scenarios.get("clean_fedavg", {}).get("test_metrics", {})
    attacked = scenarios.get("attacked_fedavg", {}).get("test_metrics", {})
    defended = scenarios.get("defended_fedavg", {}).get("test_metrics", {})

    rejected = sum(1 for r in audit if not r.get("accepted", False))
    accepted = sum(1 for r in audit if r.get("accepted", False))
    malicious_rejected = sum(
        1 for r in audit
        if r.get("attack_injected") and not r.get("accepted", False)
    )
    malicious_attempts = sum(1 for r in audit if r.get("attack_injected"))
    legitimate_rejected = sum(
        1 for r in audit
        if not r.get("attack_injected") and not r.get("accepted", False)
    )

    cards = "".join([
        metric_card("Clean F1-score", fmt(clean.get("f1")), "Held-out test set"),
        metric_card("Attacked F1-score", fmt(attacked.get("f1")), "Amplified update attack"),
        metric_card("Defended F1-score", fmt(defended.get("f1")), "After update screening"),
        metric_card("Updates rejected", rejected, "Across defended rounds"),
    ])

    table_rows = ""
    for key, (label, kind) in SCENARIOS.items():
        result = scenarios.get(key, {})
        m = result.get("test_metrics", {})
        threshold = result.get("selected_threshold", "—")
        table_rows += f"""
        <tr>
          <td><span class="pill {kind.lower()}">{escape(label)}</span></td>
          <td>{escape(fmt(threshold, 2))}</td>
          <td>{escape(pct(m.get('precision')))}</td>
          <td>{escape(pct(m.get('recall')))}</td>
          <td><b>{escape(fmt(m.get('f1')))}</b></td>
          <td>{escape(str(m.get('tp', '—')))}</td>
          <td>{escape(str(m.get('fp', '—')))}</td>
          <td>{escape(str(m.get('fn', '—')))}</td>
          <td>{escape(str(m.get('tn', '—')))}</td>
        </tr>"""

    client_stats = defaultdict(lambda: {
        "accepted": 0, "rejected": 0, "attack": 0, "reasons": []
    })
    for row in audit:
        client = row.get("client", "unknown")
        if row.get("accepted"):
            client_stats[client]["accepted"] += 1
        else:
            client_stats[client]["rejected"] += 1
        if row.get("attack_injected"):
            client_stats[client]["attack"] += 1
        reason = row.get("reason")
        if reason and reason != "accepted":
            client_stats[client]["reasons"].append(reason)

    client_rows = ""
    for client in ("client_a", "client_b", "client_c"):
        s = client_stats[client]
        if s["rejected"]:
            status = "QUARANTINED UPDATE"
            cls = "bad"
        else:
            status = "NO REJECTIONS"
            cls = "good"
        client_rows += f"""
        <tr>
          <td><b>{escape(client)}</b></td>
          <td>{s['accepted']}</td>
          <td>{s['rejected']}</td>
          <td>{s['attack']}</td>
          <td><span class="pill {cls}">{status}</span></td>
          <td>{escape(', '.join(sorted(set(s['reasons']))) or '—')}</td>
        </tr>"""

    audit_rows = ""
    for row in reversed(audit[-30:]):
        accepted_flag = bool(row.get("accepted"))
        status = "ACCEPTED" if accepted_flag else "REJECTED"
        cls = "good" if accepted_flag else "bad"
        audit_rows += f"""
        <tr>
          <td>{escape(str(row.get('round', '—')))}</td>
          <td>{escape(str(row.get('client', '—')))}</td>
          <td>{escape(fmt(row.get('update_norm'), 4))}</td>
          <td>{escape(fmt(row.get('norm_limit'), 4))}</td>
          <td><span class="pill {cls}">{status}</span></td>
          <td>{escape(str(row.get('reason', '—')))}</td>
        </tr>"""

    updated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    no_data = ""
    if not scenarios:
        no_data = """
        <div class="notice">
          No experiment results found. Run
          <code>python scripts/banksim/simulate_defense.py</code>
          and refresh this page.
        </div>"""

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="refresh" content="15">
<title>FederatedShield AI</title>
<style>
:root {{
  color-scheme: dark; --bg:#0b1020; --panel:#121b30; --line:#293651;
  --text:#eef3ff; --muted:#a5b3cb; --cyan:#55d9f3;
  --green:#64e6ae; --red:#ff7f93; --amber:#ffd37a;
}}
* {{box-sizing:border-box}}
body {{margin:0;background:var(--bg);color:var(--text);
  font:15px/1.5 system-ui,-apple-system,Segoe UI,sans-serif}}
main {{max-width:1400px;margin:auto;padding:28px 20px 50px}}
header {{display:flex;justify-content:space-between;align-items:center;
  gap:16px;flex-wrap:wrap;margin-bottom:24px}}
h1 {{font-size:clamp(26px,4vw,40px);margin:0;letter-spacing:-1px}}
h2 {{font-size:20px;margin:0 0 14px}}
h3 {{font-size:16px;margin:0 0 10px}}
p {{margin:8px 0}}
.muted {{color:var(--muted)}} .small {{font-size:12px}}
.brand {{color:var(--cyan);font-weight:800;letter-spacing:2px;font-size:12px}}
.status {{border:1px solid var(--line);padding:8px 12px;border-radius:20px;
  color:var(--green);font-size:12px}}
.grid {{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:14px}}
.card,.panel {{background:var(--panel);border:1px solid var(--line);
  border-radius:15px;padding:18px;min-width:0}}
.big {{font-size:30px;font-weight:800;letter-spacing:-1px;margin:7px 0}}
.section {{margin-top:26px}}
.panel {{overflow:hidden}}
table {{border-collapse:collapse;width:100%;font-size:13px}}
th,td {{padding:12px 10px;border-bottom:1px solid var(--line);
  text-align:left;white-space:nowrap}}
th {{color:var(--muted);font-weight:600;background:#10182b}}
tr:last-child td {{border-bottom:0}}
.table-wrap {{overflow-x:auto}}
.pill {{display:inline-block;border-radius:20px;padding:4px 8px;
  font-size:11px;font-weight:800;letter-spacing:.3px}}
.baseline {{background:#23334d;color:var(--cyan)}}
.attack,.bad {{background:#482333;color:var(--red)}}
.defended,.good {{background:#173d35;color:var(--green)}}
.notice {{border:1px solid #715d2e;background:#2d2719;padding:14px;
  border-radius:10px;margin:16px 0;color:var(--amber)}}
.two {{display:grid;grid-template-columns:1fr 1fr;gap:16px}}
.bar-row {{margin:14px 0}}
.bar-label {{display:flex;justify-content:space-between;gap:10px;margin-bottom:5px}}
.track {{height:9px;background:#28334b;border-radius:10px;overflow:hidden}}
.fill {{height:100%;background:var(--cyan);border-radius:10px}}
.fill.red {{background:var(--red)}} .fill.green {{background:var(--green)}}
footer {{margin-top:26px;border-top:1px solid var(--line);padding-top:18px;
  color:var(--muted);font-size:12px}}
code {{overflow-wrap:anywhere;color:var(--cyan)}}
@media(max-width:850px) {{.grid{{grid-template-columns:repeat(2,minmax(0,1fr))}}
  .two{{grid-template-columns:1fr}}}}
@media(max-width:450px) {{.grid{{grid-template-columns:1fr}} main{{padding:18px 12px}}}}
</style>
</head>
<body><main>
<header>
 <div><div class="brand">CYBERSECURITY / FEDERATED AI</div>
 <h1>FederatedShield AI</h1>
 <p class="muted">Fraud detection · Client trust · Selective containment</p></div>
 <div class="status">● LOCAL DEMO DASHBOARD</div>
</header>
{no_data}
<div class="grid">{cards}</div>

<section class="section panel">
 <h2>Model performance comparison</h2>
 <p class="muted small">Held-out BankSim test set. Each scenario selects its threshold using validation data.</p>
 <div class="table-wrap"><table>
 <thead><tr><th>Scenario</th><th>Threshold</th><th>Precision</th><th>Recall</th>
 <th>F1</th><th>TP</th><th>FP</th><th>FN</th><th>TN</th></tr></thead>
 <tbody>{table_rows}</tbody></table></div>
 <div class="two section">
  <div><h3>F1-score</h3>
   {''.join(f'<div class="bar-row"><div class="bar-label"><span>{escape(label)}</span><b>{escape(fmt(scenarios.get(key, {}).get("test_metrics", {}).get("f1")))}</b></div><div class="track"><div class="fill {"red" if key == "attacked_fedavg" else "green" if key == "defended_fedavg" else ""}" style="width:{max(0,min(100,float(scenarios.get(key, {}).get("test_metrics", {}).get("f1",0))*100)):.2f}%"></div></div></div>' for key,(label,kind) in SCENARIOS.items())}
  </div>
  <div><h3>Attack containment summary</h3>
   <p>Injected malicious updates: <b>{malicious_attempts}</b></p>
   <p>Malicious updates rejected: <b>{malicious_rejected}</b></p>
   <p>All updates rejected: <b>{rejected}</b></p>
   <p>Legitimate updates rejected: <b>{legitimate_rejected}</b></p>
   <p class="muted small">Acceptance count: {accepted}. Counts cover the defended scenario's audit log.</p>
  </div>
 </div>
</section>

<section class="section panel">
 <h2>Client trust and containment</h2>
 <p class="muted small">Status reflects update decisions in the recorded controlled experiment, not a permanent judgment about a real institution.</p>
 <div class="table-wrap"><table>
 <thead><tr><th>Client</th><th>Accepted</th><th>Rejected</th><th>Attack attempts</th><th>Observed status</th><th>Rejection reason</th></tr></thead>
 <tbody>{client_rows}</tbody></table></div>
</section>

<section class="section panel">
 <h2>Recent security audit events</h2>
 <p class="muted small">Latest 30 recorded decisions, newest first.</p>
 <div class="table-wrap"><table>
 <thead><tr><th>Round</th><th>Client</th><th>Update norm</th><th>Norm limit</th><th>Decision</th><th>Reason</th></tr></thead>
 <tbody>{audit_rows or '<tr><td colspan="6">No audit events found.</td></tr>'}</tbody>
 </table></div>
</section>

<section class="section notice">
 <b>Research prototype — limitations</b>
 <p>This experiment uses synthetic BankSim data and three simulated clients in one process.
 The demonstrated norm check catches a deliberately amplified update; it does not detect all
 poisoning attacks. This dashboard does not provide client isolation, differential privacy,
 or production-grade identity verification.</p>
</section>
<footer>
 FederatedShield AI · Data read from local experiment outputs · Page refreshes every 15 seconds<br>
 Generated at {escape(updated)} · IBM LinuxONE host
</footer>
</main></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.split("?", 1)[0].split("#", 1)[0] == "/favicon.ico":
            self.send_response(204)
            self.end_headers()
            return
        request_path = self.path.split("?", 1)[0].split("#", 1)[0]
        if request_path not in ("/", "/index.html", "/**"):
            self.send_error(404)
            return
        page = render().encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(page)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(page)

    def log_message(self, fmt_string, *args):
        print(f"[dashboard] {self.address_string()} - {fmt_string % args}")


if __name__ == "__main__":
    print(f"FederatedShield AI dashboard on http://127.0.0.1:{PORT}")
    print("Keep this process running while viewing the dashboard.")
    HTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
