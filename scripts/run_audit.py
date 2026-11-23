from tokentax_lab.audit.flores import load_local_flores
from tokentax_lab.audit.engine import AuditEngine
from tokentax_lab.data import save_json

texts = load_local_flores("data/flores", ("kn", "hi", "ta", "en"))
engine = AuditEngine()
rows = engine.audit_many(("cl100k_base", "o200k_base"), texts)
rows += engine.audit_hf("Qwen/Qwen2.5-0.5B", texts, "Qwen2.5")
save_json("results/audit.json", rows)
print("Wrote results/audit.json")
