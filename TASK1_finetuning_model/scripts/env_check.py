"""Phase 0 environment verification — run on the remote Colab GPU kernel."""
from __future__ import annotations

import json
import platform

from common import TRACK_DIR, set_seed

def main() -> int:
    set_seed()
    info: dict = {
        "seed": 42,
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        info["cuda_version"] = torch.version.cuda
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            info["vram_gb"] = round(props.total_memory / 1e9, 2)
    except Exception as exc:  # noqa: BLE001
        info["torch_error"] = str(exc)

    for pkg in ("transformers", "peft", "datasets", "accelerate", "bitsandbytes"):
        try:
            mod = __import__(pkg)
            info[pkg] = getattr(mod, "__version__", "?")
        except ImportError:
            info[pkg] = None

    out = TRACK_DIR / "logs" / "env_check.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(info, indent=2))
    if not info.get("cuda_available"):
        print("WARNING: CUDA not available - a GPU is required (Phase 0 DoD).")
        return 1
    print("OK - environment verified, logged to logs/env_check.json")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
