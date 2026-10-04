from pathlib import Path
import sys
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, 'reconfigure'):
        stream.reconfigure(encoding='utf-8', errors='replace')
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
