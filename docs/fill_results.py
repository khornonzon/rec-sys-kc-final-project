"""Вставляет таблицу результатов симулятора в README между маркерами RESULTS."""

import json
import sys
from pathlib import Path

result = json.loads(Path(sys.argv[1]).read_text())
cfg = result["config"]
lines = [
    f"Прогон: {cfg['users']} пользователей × {cfg['rounds']} раундов, "
    f"каталог {cfg['items']} объектов, до {cfg['concurrency']} параллельных пользователей, "
    f"Docker Desktop на ноутбуке (`docs/simulation_result.json`).",
    "",
    "| Раунд | precision@10 | ndcg@10 | diversity | coverage | p50, мс | p95, мс |",
    "|------:|-------------:|--------:|----------:|---------:|--------:|--------:|",
]
for r in result["rounds"]:
    lines.append(
        f"| {r['round']} | {r['precision']:.3f} | {r['ndcg']:.3f} | {r['diversity']:.3f} | "
        f"{r['coverage']:.3f} | {r['latency_p50_ms']:.0f} | {r['latency_p95_ms']:.0f} |"
    )
lines.append(
    f"| **итого** | **{result['precision']:.3f}** | **{result['ndcg']:.3f}** | "
    f"**{result['diversity']:.3f}** | **{result['coverage']:.3f}** | "
    f"**{result['latency_p50_ms']:.0f}** | **{result['latency_p95_ms']:.0f}** |"
)
lines += [
    "",
    "Латентность здесь измерена клиентом вместе с сетью Docker Desktop и очередью из "
    "50 параллельных запросов (клиент на Python сам упирается в CPU); серверная гистограмма "
    "`recs_request_seconds` показывает среднее около 1 мс, см. заметку о производительности ниже.",
]
block = "\n".join(lines)
readme = Path("README.md")
text = readme.read_text()
start = text.index("<!-- RESULTS:BEGIN -->") + len("<!-- RESULTS:BEGIN -->")
end = text.index("<!-- RESULTS:END -->")
readme.write_text(text[:start] + "\n" + block + "\n" + text[end:])
print(block)
