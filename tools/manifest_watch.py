"""Diff the public-circuit manifest against our deployed portfolio.

Downloads manwithacat's manifest.csv and reports per-task the best
public artifact vs what we currently deploy. Run periodically; when a
better artifact appears, fetch and verify it.

Usage:
    ../.venv/bin/python tools/manifest_watch.py
"""
import csv, io, os, subprocess, sys

MANIFEST_DS = 'manwithacat/vqcc-apache-public-circuits'
DEPLOYED = {
    'invsqrt': ('ecf908fec987', 74383.66),
    'dqi': ('4471b4933c18', 345236065276.0),
    'dfthcsos': (None, 294642170.6),
}


def fetch_manifest():
    out = subprocess.run(
        ['./.venv/bin/kaggle', 'datasets', 'download', '-d', MANIFEST_DS,
         '-f', 'manifest.csv', '-p', '/tmp', '--force'],
        capture_output=True, text=True, cwd=os.path.dirname(os.path.dirname(__file__)))
    path = '/tmp/manifest.csv'
    if not os.path.exists(path):
        import zipfile
        zf = os.path.join('/tmp', 'manifest.csv.zip')
        if os.path.exists(zf):
            with zipfile.ZipFile(zf) as z:
                z.extractall('/tmp')
    return path


def main():
    path = fetch_manifest()
    rows = list(csv.DictReader(open(path)))
    for task, (sha, cost) in DEPLOYED.items():
        cand = [r for r in rows if r['task'] == task and r.get('local_task_score')]
        cand.sort(key=lambda r: float(r['local_task_score']))
        if not cand:
            print('%s: no scored artifacts' % task)
            continue
        best = cand[0]
        best_cost = float(best['local_task_score'])
        marker = '  <-- NEW BEST' if best['sha12'] != sha and cost and best_cost < cost else ''
        print('%s: best public %s cost %.4g | deployed %s cost %.4g%s'
              % (task, best['sha12'], best_cost, sha, cost or -1, marker))
        print('     source: %s' % best.get('source_notebook_url', ''))


if __name__ == '__main__':
    main()
