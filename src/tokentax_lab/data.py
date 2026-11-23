from pathlib import Path
import json
import yaml

ROOT = Path(__file__).resolve().parents[2]

def load_config(path: str | Path | None = None) -> dict:
    path = Path(path) if path else ROOT / "config.yaml"
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def save_json(path: str | Path, payload) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

def load_json(path: str | Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))
