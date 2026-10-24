# MonoLM: болгарская LLM с нуля

Домашнее задание №1: подготовка корпуса, свой токенизатор и pretrain decoder-only модели. **Полное обучение пока не запускалось**; в репозитории — код, выполненные отчёты и короткие проверки.

Модель: 346 096 576 параметров, 32 слоя, hidden 960, FFN 2560, GQA 15/5, RMSNorm + QK-Norm, SwiGLU, RoPE, контекст 2048, связанные embeddings, BPE-32768. `Qwen3ForCausalLM` используется как реализация архитектуры; все веса создаются случайно.

## С нуля на L40

Нужны **Linux x86_64, Python 3.12, L40 и драйвер с поддержкой CUDA 12.8**. Ориентир: 32 ГБ RAM и 100 ГБ свободного диска; около 200 ГБ, если также воспроизводить сравнение с raw FineWeb-2. Подготовка данных не входит в суточный бюджет обучения.

Профиль L40: **PyTorch 2.8.0 / CUDA 12.8 / Flash Attention 2.8.3.post2**, совместимые официальные wheels для Python 3.12, без сборки CUDA из исходников. L40 относится к Ada и имеет 48 ГБ VRAM. Отдельный профиль `5080` сохраняет проверенную связку Torch 2.14.1 / CUDA 13 / FA4. На самой L40 замеров пока не было: установка завершается проверкой Flash Attention forward/backward. [NVIDIA L40](https://www.nvidia.com/en-us/data-center/l40/), [Flash Attention](https://github.com/Dao-AILab/flash-attention).

Все следующие команды выполняются из `task1/`:

```bash
cd task1
```

```bash
# Окружение в ../venv и проверка GPU.
bash scripts/setup.sh l40

# HQ/test → Top 1M → обучение BPE-32k → token streams.
bash scripts/pipeline.sh prepare

# Проверки и подбор batch/длительности именно на этой GPU.
bash scripts/pipeline.sh calibrate l40

# Отдельная явная команда полного обучения.
bash scripts/train.sh
```

`prepare` и `calibrate` не запускают финальное обучение. На другой GPU заново выполните калибровку: старое время RTX 5080 не является оценкой для L40. `train` проверяет соответствие GPU сохранённой калибровке.

Посмотреть команды без выполнения: `DRY_RUN=1 bash scripts/pipeline.sh prepare`. Этот режим есть у всех shell-скриптов. Для долгого запуска используйте `tmux new -s monolm`, затем `bash scripts/train.sh`; отсоединиться — `Ctrl+B`, затем `D`.

## Структура и пути

```text
task1/
  scripts/                   setup, pipeline, train, Trackio
  configs/                   конфиги L40 и RTX 5080
  requirements/              закреплённые зависимости по профилям
  src/
    data/                    загрузка, отбор, статистика
    tokenization/            обучение и сравнение токенизаторов
    training/                модель, данные, цикл, чекпоинты, проверки
    evaluation/              слепой аудит
    inference.py             генерация из локального экспорта
    environment.py           диагностика окружения
  notebooks/                 выполненные отчёты
  artifacts/                 данные и результаты; исключены из Git
```

Код устанавливается editable из `task1/pyproject.toml`. Python-пакет — `src`, без дополнительной вложенной папки. CLI: `../venv/bin/python -m src --help`. Shell-скрипты работают независимо от текущей директории.

По умолчанию артефакты находятся в `artifacts/`. Для диска виртуалки задайте путь **перед всеми этапами**, включая dashboard:

```bash
export MONOLM_ARTIFACTS=/data/monolm
bash scripts/pipeline.sh prepare
bash scripts/pipeline.sh calibrate l40
bash scripts/train.sh
```

Также доступны `MONOLM_VENV` (окружение), `PYTHON_BIN` (Python 3.12 для установки), `MONOLM_RAW_DATA` (большой raw-корпус). FA2 и FA4 используют общее пространство имён `flash_attn`: для смены профиля создайте отдельный `MONOLM_VENV`.

| Путь внутри артефактов | Содержимое |
|---|---|
| `sources/` | Закреплённые версии исходных parquet |
| `corpus/train/` | 10 файлов по 100 000 документов |
| `corpus/{validation,test}.parquet` | По 5000 документов |
| `tokenizers/selected/` | HF-совместимый токенизатор |
| `training/data/` | uint16-потоки, границы документов, manifest |
| `training/recommended_config.json` | Конфиг после замера текущей GPU |
| `training/runs/monolm-346m/` | Финальный запуск, чекпоинты, JSONL и экспорт |
| `training/checks/` | Проверочные веса; не использовать для финального старта |
| `training/trackio/` | Локальные логи |

Загрузка использует кеш Hugging Face. Готовые корпус и token streams проверяются по SHA-256; завершённые токенизаторы повторно не обучаются. Если manifest ещё не создан, незавершённый этап выполняется заново. Для нового эксперимента с изменёнными входами/версиями используйте отдельный каталог артефактов.

## Данные и эксперименты

Основной путь скачивает [HQ-Plus](https://huggingface.co/datasets/epfml/FineWeb2-HQ-Plus) и официальный болгарский test [FineWeb-2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2). Raw train для обучения не нужен. Commit SHA закреплены в коде:

- FineWeb-2: `af9c13333eb981300149d5ca60a8e9d659b276b9`.
- HQ-Plus: `146690ef88e86e6d728a03ea6f6d85037d8cf9fa`.

Удаляются пустые тексты, точные дубликаты и пересечения train/holdout по ID или хешу. Для хеша — NFC, нормализация переводов строк и внешних пробелов; исходный текст сохраняется. Отбор — опубликованный score по убыванию, ID по возрастанию, затем перемешивание seed 42. Классификатор повторно не запускается.

`prepare` заново обучает **фиксированный BPE-32768**, выбранный ранее, на воспроизводимых 50 000 документах. Это не новый поиск победителя. Повторить BPE/Unigram × 8k/16k/32k/64k:

```bash
bash scripts/pipeline.sh compare-tokenizers
```

Этот этап создаёт полный токенизаторный отчёт. Если победит другой размер словаря, согласуйте его с моделью и пересоздайте token streams в отдельном каталоге.

Тексты кодируются без обрезания, после каждого добавляется EOS ID 3. Обучение — блоки по 2048 токенов со сдвигом цели на один; attention переходит через EOS. Последний блок дополняется PAD с исключением дополнения из loss. В выполненной подготовке получилось **1 204 544 281 токен train с EOS**; точное число нового запуска записывается в manifest. Test не используется при обучении токенизатора или LLM.

Сравнение с raw FineWeb-2 и подготовка слепого аудита — отдельный этап:

```bash
bash scripts/pipeline.sh corpus-analysis
```

Он дополнительно скачивает около 46 ГиБ raw train, готовит три выборки по 10 000 документов, статистики и пакеты. **Аудит требует внешних оценок**: скрипт их не выдумывает и не вызывает платный API. Формат и рубрика находятся в пакетах. После оценивания и разбора расхождений: `../venv/bin/python -m src audit summarize`; прежняя строгая рубрика доступна через `--rubric strict`. Полный dataset-ноутбук использует обе рубрики. Высокий HQ-score не является доказательством улучшения обученной LLM.

## Обучение и восстановление

Accelerate, BF16 autocast, FP32 веса/градиенты/состояния AdamW, clipping 1.0, peak LR 3e-4, warmup 2%, cosine до 3e-5. Эффективный batch — 131 072 токена. Микробатч и activation checkpointing подбираются замером. Лимит — одна эпоха и сутки; план оставляет 10% запаса. Это оценка, не гарантия времени.

Чекпоинты сохраняются каждые 15 минут, при улучшении validation loss и завершении. В них — веса, optimizer, scheduler, RNG, позиция в данных, счётчики и время. Запись атомарная, с fsync и SHA-256; сохраняются два последних и лучший. Resume выполняйте **новым процессом**:

```bash
bash scripts/train.sh --resume artifacts/training/runs/monolm-346m/checkpoints/last.json
```

При `MONOLM_ARTIFACTS` замените путь. Несовпадение конфига, данных, токенизатора, исходников или зависимостей вызывает явную ошибку. Проверочные чекпоинты до реорганизации кода остаются историческими артефактами.

Экспорт лучшего чекпоинта и инференс:

```bash
../venv/bin/python -m src export \
  --resume artifacts/training/runs/monolm-346m/checkpoints/best.json \
  --run artifacts/training/export-best

../venv/bin/python -m src infer \
  --model artifacts/training/export-best \
  --prompt 'България е страна' --max-new-tokens 128
```

Это простой pretrain без чат-шаблона: prompt — начало текста. Экспорт поддерживает `AutoModelForCausalLM.from_pretrained()` и `AutoTokenizer.from_pretrained()`.

## Trackio и ноутбуки

Локальный dashboard: `bash scripts/trackio.sh`. Для VM можно пробросить порт: `ssh -L 7860:127.0.0.1:7860 user@vm`. Фактический порт печатает Trackio. Метрики дублируются в JSONL; dashboard не требуется держать открытым. Публикации в Hugging Face Space нет.

- [Корпус и аудит](notebooks/dataset.ipynb).
- [Сравнение токенизаторов](notebooks/tokenizer.ipynb).
- [Модель и замеры RTX 5080](notebooks/train.ipynb).

Сохранённые результаты относятся к прежним экспериментам. На новой машине train-ноутбук выполняется после `calibrate`, токенизаторный — после `compare-tokenizers`, dataset — после статистики и обеих рубрик аудита. Например:

```bash
../venv/bin/python -m jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=180 notebooks/train.ipynb
```

Данные, окружение и локальный AGENTS.md исключены из Git. Коммиты автоматически не создаются: перед клонированием на VM добавьте новые исходники в свой коммит либо перенесите рабочую копию. Артефакты и проверочные веса для запуска с нуля не нужны.
