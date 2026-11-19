"""Assemble a submission.zip from chosen components.

Usage:
    ../.venv/bin/python tools/pack_submission.py OUT_DIR \
        --invsqrt PATH --dqi PATH --dfthcsos DIR

The archive has invsqrt.kmb, dqi.kmb, dfthcsos/ at the root, as required.
"""
import argparse, os, shutil, subprocess, sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument('out_dir')
    p.add_argument('--invsqrt', required=True)
    p.add_argument('--dqi', required=True)
    p.add_argument('--dfthcsos', required=True)
    a = p.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    shutil.copy(a.invsqrt, os.path.join(a.out_dir, 'invsqrt.kmb'))
    shutil.copy(a.dqi, os.path.join(a.out_dir, 'dqi.kmb'))
    dst = os.path.join(a.out_dir, 'dfthcsos')
    if os.path.isdir(dst):
        shutil.rmtree(dst)
    shutil.copytree(a.dfthcsos, dst)

    zpath = os.path.join(a.out_dir, 'submission.zip')
    if os.path.exists(zpath):
        os.remove(zpath)
    subprocess.run(['zip', '-qr', 'submission.zip', 'invsqrt.kmb', 'dqi.kmb',
                    'dfthcsos'], cwd=a.out_dir, check=True)
    subprocess.run(['unzip', '-l', 'submission.zip'], cwd=a.out_dir)
    print('\nTo submit:')
    print('  ./.venv/bin/kaggle competitions submit '
          '-c google-verifiable-quantum-circuit-compilation '
          f'-f {zpath} -m "msg"')


if __name__ == '__main__':
    main()
