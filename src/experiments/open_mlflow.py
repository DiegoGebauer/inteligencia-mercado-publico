"""Muestra el registro local del Proyecto 2 en http://127.0.0.1:5000."""
import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def server_command(root, port=5000):
    database = root / 'mlflow_local/mlflow.db'
    if not database.is_file():
        raise FileNotFoundError('Primero ejecuta python -m src.experiments.register_affinity_baseline.')
    return [sys.executable, '-m', 'mlflow', 'server', '--backend-store-uri', 'sqlite:///' + database.resolve().as_posix(), '--no-serve-artifacts', '--host', '127.0.0.1', '--port', str(port)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=5000)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error('El puerto debe estar entre 1 y 65535.')
    if importlib.util.find_spec('mlflow') is None:
        raise RuntimeError('Instala primero requirements-mlflow.txt con constraints-mlflow.txt.')
    command = server_command(ROOT, args.port)
    print(f'Abre http://127.0.0.1:{args.port} en tu navegador. Mantén esta terminal abierta; Ctrl+C detiene el servidor.', flush=True)
    try:
        subprocess.run(command, cwd=ROOT, check=True)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
