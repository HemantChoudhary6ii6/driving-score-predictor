"""
run_pipeline.py  —  Deep Learning Pipeline Runner
===================================================
Runs all three steps in order:
  Step 0: Prepare data sequences
  Step 1: Train LSTM, GRU, Transformer
  Step 2: Evaluate and generate charts

Usage:
  python run_pipeline.py

After completion:
  python step3_score.py path/to/session.csv   ← score a new session
  streamlit run app.py                         ← launch dashboard
"""

import subprocess, sys, os


def run(script, label):
    print(f"\n{'='*55}")
    print(f"  {label}")
    print(f"{'='*55}")
    r = subprocess.run([sys.executable, script])
    if r.returncode != 0:
        print(f"\n  ✗ FAILED at {script}")
        print(f"    Fix the error above, then re-run: python {script}")
        sys.exit(1)
    print(f"  ✓ Done")


def main():
    for d in ['data', 'models', 'outputs']:
        os.makedirs(d, exist_ok=True)

    run('step0_prepare.py',  'Step 0 — Prepare data sequences')
    run('step1_train.py',    'Step 1 — Train LSTM / GRU / Transformer')
    run('step2_evaluate.py', 'Step 2 — Evaluate models + generate charts')

    print(f"\n{'='*55}")
    print("  PIPELINE COMPLETE")
    print(f"{'='*55}")
    print("\n  Score a new session:")
    print("    python step3_score.py path/to/session.csv")
    print("\n  Launch dashboard:")
    print("    streamlit run app.py")
    print(f"{'='*55}\n")


if __name__ == '__main__':
    main()