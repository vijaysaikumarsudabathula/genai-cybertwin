"""
run_pipeline.py
Runs the complete GenAI-CyberTwin pipeline end-to-end, in order. Each step
writes real files under data/reports/, models/, results/, and dashboard/.
Run from the project root: `python run_pipeline.py`
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"

STEPS = [
    ("Validating dataset", "validation.py"),
    ("Writing feature dictionary", "feature_engineering.py"),
    ("Training & comparing models", "train.py"),
    ("Evaluating final model", "evaluate.py"),
    ("Running SHAP explainability", "explainability.py"),
    ("Computing cyber risk scores", "risk_engine.py"),
    ("Running counterfactual engine", "counterfactual.py"),
    ("Running ablation study", "ablation.py"),
    ("Running error analysis", "error_analysis.py"),
    ("Generating GenAI narratives & grounding tests", "generate_genai_narratives.py"),
    ("Exporting model to JSON for the dashboard", "export_model_js.py"),
    ("Assembling dashboard data", "prepare_dashboard_data.py"),
    ("Generating report figures", "make_report_figures.py"),
    ("Building the final PDF report", "report_generator.py"),
]


def main():
    total = len(STEPS)
    for i, (label, script) in enumerate(STEPS, 1):
        print(f"[{i}/{total}] {label} ({script})")
        result = subprocess.run([sys.executable, str(SRC / script)], cwd=str(SRC))
        if result.returncode != 0:
            print(f"\nPipeline stopped: {script} exited with code {result.returncode}")
            sys.exit(result.returncode)
    print("\nPipeline complete. Build the final HTML dashboard with:")
    print("  cd dashboard && python3 build_dashboard.py")


if __name__ == "__main__":
    main()
