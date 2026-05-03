# DDW SDK v0.1 — Track 1 deliverable

Это **отдельный репо** `ddw-sdk-py/`, который кладётся **рядом** с `data-decision-layer/`,
а не внутрь.

Рекомендуемая структура на твоей машине:

```
C:\Users\Achilles\Desktop\
├── data-decision-layer\       ← backend (уже есть)
└── ddw-sdk-py\                ← новый, из этого ZIP
```

## Установка

```powershell
# из папки ddw-sdk-py
conda activate decision
pip install -e ".[dev]"
```

`-e` = editable mode: правки в коде SDK сразу видны без переустановки.
`[dev]` = подтянуть pytest, ruff, mypy.

## Прогнать тесты

```powershell
cd C:\Users\Achilles\Desktop\ddw-sdk-py
pytest -q
```

Должно быть **30 зелёных тестов** (все на моках, backend не нужен).

Также:
```powershell
ruff check .
mypy ddw
```

## Локальный e2e-тест против backend

В одном терминале — стартуй backend:

```powershell
cd C:\Users\Achilles\Desktop\data-decision-layer
conda activate decision
uvicorn api.main:app --host 127.0.0.1 --port 8000
```

В другом терминале — гоняй пример:

```powershell
cd C:\Users\Achilles\Desktop\ddw-sdk-py
conda activate decision
$env:DDW_BASE_URL = "http://127.0.0.1:8000"
$env:DDW_TENANT_ID = "team-vision"

python examples\basic_sklearn.py
```

Должно создать 4 runs. Проверка:

```powershell
curl -H "X-Tenant-Id: team-vision" "http://127.0.0.1:8000/api/v1/runs"
```

Если видишь 4 объекта — SDK работает end-to-end.

Затем прогон analytics + сохранение DataPotential:

```powershell
curl -X POST `
  -H "X-Tenant-Id: team-vision" `
  -H "Content-Type: application/json" `
  "http://127.0.0.1:8000/api/v1/analytics/save-potential" `
  -d '{"model_tag":"logreg","metric_name":"f1","x_kind":"dataset_size","split":"val"}'
```

Должно вернуть DataPotential с `evidence_run_ids` указывающими на твои 4 runs.

**Это первый раз, когда полный MVP vertical работает end-to-end через продуктовый
SDK.** Backend → analytics → save → потенциал.

## Что в v0.1

- `Run` context manager — главное API
- `Client` — низкоуровневый HTTP клиент
- `Config` — env-or-args конфиг
- 4 типа исключений: `DDWError`, `DDWConfigError`, `DDWAPIError`, `DDWUsageError`
- 30 тестов, все проходят без backend
- `on_error="warn"` для production training (failures не крашат training)
- 2 примера: PyTorch-style, sklearn-style

## Что НЕ в v0.1 (намеренно)

- Time-series метрики (только `metric_final`)
- Артефакты моделей
- Background threads, retries, очереди
- Auto-instrumentation
- Async API
- W&B / MLflow integration (это Track 2/3)

## Дальше

После того как pytest зелёный и e2e-проверка прошла — переходим к **Track 2: UI**
(один Streamlit экран для analytics workflow). Backend и SDK уже готовы.

После UI — **Track 3: MLflow importer** (поверх SDK + новые поля
`external_source`/`external_id` в `RunRow`).
