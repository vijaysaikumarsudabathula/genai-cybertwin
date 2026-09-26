"""
build_dashboard.py
Injects dashboard_data.json and the exported model JSON into template.html
to produce the final, self-contained cybertwin_dashboard.html (no server,
no external files needed other than the Plotly CDN script for charts).
Run from dashboard/: `python3 build_dashboard.py`
"""
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODELS_DIR = HERE.parent / "models"


def build():
    tpl = (HERE / "template.html").read_text(encoding="utf-8")
    data_json = (HERE / "dashboard_data.json").read_text(encoding="utf-8")
    model_json = (MODELS_DIR / "xgb_model_export.json").read_text(encoding="utf-8")

    data_json_safe = data_json.replace("</script>", "<\\/script>")
    model_json_safe = model_json.replace("</script>", "<\\/script>")

    out = tpl.replace("__DASHBOARD_DATA__", data_json_safe).replace("__MODEL_DATA__", model_json_safe)
    out_path = HERE / "cybertwin_dashboard.html"
    out_path.write_text(out, encoding="utf-8")
    print(f"Wrote {out_path} ({out_path.stat().st_size / 1024 / 1024:.2f} MB)")


if __name__ == "__main__":
    build()
