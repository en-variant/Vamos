# Data layout

Place user-provided or downloaded public assets here.

- `flores/kn.devtest`, `flores/hi.devtest`, `flores/ta.devtest`, `flores/en.devtest`
- `belebele/<language>.jsonl` or the exact Hugging Face dataset loaded in a concrete script
- `indic/kn.txt`, `indic/hi.txt`, `indic/ta.txt`: held-out text, disjoint from FLORES

Never merge held-out adaptation text with FLORES devtest.
