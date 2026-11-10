import json, pathlib, csv
here = pathlib.Path(__file__).parent
d1 = json.load(open(here / "results.json")); d2 = json.load(open(here / "results_v2.json")); d3 = json.load(open(here / "results_v3.json")); d4 = json.load(open(here / "evaluation_report.json"))
rows = list(csv.DictReader(open(here / "sample_output_v3.csv")))[::3]
num = ["speed_kmh","risk_next_60s","risk_low","risk_high","safety_index","p_minor_or_worse","p_major_or_worse","p_critical"]
for r in rows:
    for k in num: r[k] = float(r[k])
d3["sample"] = rows
html = (here / "template.html").read_text().replace("__DATA4__", json.dumps(d4, default=float)).replace("__DATA3__", json.dumps(d3, default=float)).replace("__DATA2__", json.dumps(d2, default=float)).replace("__DATA__", json.dumps(d1, default=float))
(here / "ligtas-cascade.html").write_text(html)
print("wrote", here / "ligtas-cascade.html", len(html) // 1024, "KB")
