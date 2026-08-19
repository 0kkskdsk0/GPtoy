"""pytest 公共夹具：确保仓库根目录在 sys.path 上（未 editable install 时也能 import gptoy）。"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
