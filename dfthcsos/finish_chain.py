"""End-to-end driver: chain checkpoint -> pad B -> W-refit vs decoded words -> lambda opt
-> submission dir -> official verify.

Usage: python dfthcsos/finish_chain.py runs/chain/chk_*.npz --tag v1 [--brot 8] [--skip-verify] 
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
ROOT = Path(__file__).resolve().parents[1]
PY = ROOT / '.venv/bin/python'
D = Path(__file__).resolve().parent


def run(cmd, **kw):
    print('+', ' '.join(str(c) for c in cmd), flush=True)
    return subprocess.run([str(c) for c in cmd], check=True, **kw)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('chk', type=Path)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--brot', type=int, default=8)
    ap.add_argument('--B', type=int, default=129)
    ap.add_argument('--wrefit', type=int, default=800)
    ap.add_argument('--lam-steps', type=int, default=2000)
    ap.add_argument('--skip-verify', action='store_true')
    a = ap.parse_args()
    work = ROOT / 'runs' / a.tag
    work.mkdir(parents=True, exist_ok=True)

    padded = work / 'padded.npz'
    run([PY, D / 'pad_b.py', a.chk, padded, '--B', a.B, '--refit', 300])
    qfac = work / 'qfac.npz'
    run([PY, D / 'wrefit.py', padded, qfac, '--brot', a.brot, '--steps', a.wrefit])
    lam = work / 'lam.npz'
    run([PY, D / 'lambda_stage.py', qfac, '--out', lam, '--steps', a.lam_steps])
    out = ROOT / 'out' / a.tag
    run([PY, D / 'pipeline.py', lam, out, '--brot', a.brot, '--verify'] if not a.skip_verify
        else [PY, D / 'pipeline.py', lam, out, '--brot', a.brot])


if __name__ == '__main__':
    main()
