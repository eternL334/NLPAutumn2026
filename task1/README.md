# MonoLM: болгарская LLM с нуля

Домашнее задание №1: подготовка корпуса, свой токенизатор и pretrain decoder-only модели. Модель обучена с нуля до **5000 шагов / 655 358 744 токенов**, запуск занял 12 ч 31 мин и остановился по заданному лимиту. Validation loss на фиксированных 262 144 токенах — 3,233. Исторические ноутбуки подготовки модели содержат замеры до этого запуска.

Модель: 346 096 576 параметров, 32 слоя, hidden 960, FFN 2560, GQA 15/5, RMSNorm + QK-Norm, SwiGLU, RoPE, контекст 2048, связанные embeddings, BPE-32768. `Qwen3ForCausalLM` используется как реализация архитектуры; все веса создаются случайно.

## Быстрая проверка модели и результатов

- [Веса и токенизатор на Hugging Face](https://huggingface.co/eternl/monolm-346m-bg) — checkpoint шага 5000.
- [Отдельный ноутбук инференса](notebooks/inference.ipynb) — загружает модель с HF и содержит выполненные примеры на CPU. Его можно скачать и выполнить в Jupyter или Colab; исходный корпус и пакет `src` ему не нужны.
- [Выполненный бенчмарк](notebooks/evaluation.ipynb) — таблицы, графики, рубрика LLM-judge и примеры ответов доступны при просмотре ноутбука. Краткие результаты приведены [ниже](#бенчмарк-генерации).
- [Графики обучения Trackio](https://huggingface.co/spaces/eternl/monolm-training) — группа `monolm-346m`.

Код и отчёты находятся в этом Git-репозитории. Для инференса достаточно Python 3.12 и следующих библиотек; Flash Attention не требуется:

```bash
python -m pip install 'torch>=2.6' 'transformers==5.20.2' 'tokenizers==0.23.3' 'huggingface-hub==2.2.0' 'safetensors==0.8.0'
```

Выполните следующий Python-код. Он работает на CPU или CUDA и скачивает публичную модель без токена; первая загрузка весов — около 1,38 ГБ.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

repo = "eternl/monolm-346m-bg"
revision = "b75dfb5da41fe98b1fb669a878eb662389deb199"
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) if device == "cuda" else torch.float32
tokenizer = AutoTokenizer.from_pretrained(repo, revision=revision, token=False, local_files_only=False)
model = AutoModelForCausalLM.from_pretrained(
    repo, revision=revision, token=False, dtype=dtype, attn_implementation="sdpa"
).to(device).eval()

prompt = "Момичето отвори прозореца и"
inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
torch.manual_seed(42)
with torch.inference_mode():
    output = model.generate(
        **inputs, max_new_tokens=64, do_sample=True,
        temperature=0.8, top_p=0.95, top_k=0, use_cache=True,
        pad_token_id=0, eos_token_id=3, suppress_tokens=[0, 1, 2],
    )
print(prompt + tokenizer.decode(output[0, inputs.input_ids.shape[1]:], skip_special_tokens=True))
```

Это pretrain-модель продолжения текста без chat template. Меняйте `prompt` на начало болгарского текста; суммарная длина входа и продолжения должна помещаться в 2048 токенов. Связность и ответы на вопросы пока ограничены — измерения приведены в бенчмарке.

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

Каждый вариант калибровки запускается отдельным процессом; сохранение, resume и загрузка экспорта тоже проверяются в отдельных процессах. Профиль `5080` проверяет микробатчи 2 и 1 только с activation checkpointing и требует минимум 2 ГиБ свободной VRAM после замера. L40 сравнивает варианты с checkpointing и без него. Логи каждого процесса сохраняются в `artifacts/training/calibrations/<id>/`; `recommended_config.json` обновляется только после всех успешных проверок.

Если калибровка завершилась `CUDA driver error: device not ready` / `Failed to create GPU mapping`, это ошибка драйвера, а не успешный результат или обычный пойманный OOM. Повторите калибровку новым процессом. Если ошибка сохраняется в WSL, закройте работающие задачи WSL, выполните `wsl --shutdown` из Windows PowerShell, затем откройте WSL и повторите команду. Перезапуск WSL завершает все его процессы. Это способ восстановить окружение, а не доказанное устранение причины сбоя.

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

## Бенчмарк генерации

`bg-mini-v1` сравнивает MonoLM-346M на шаге 5000 с `Qwen/Qwen3-0.6B-Base` (revision `da87bfb608c14b7cf20ba1ce41287e8de496c0cd`). На модель приходится 60 продолжений из validation и 20 простых вопросов Belebele в двух формулировках. Это адаптированный небольшой набор, а не официальный результат Belebele. Источник QA: [facebook/belebele](https://huggingface.co/datasets/facebook/belebele), revision `dd9520af047cac89fd3b7f5d463f17144b1627a2`, лицензия CC-BY-SA-4.0; сохранены исходные вопросы, ответы, ссылки и подтверждающие фрагменты. Новые формулировки и решения отбора — в `configs/evaluation_bg_mini_v1.json`.

```bash
../venv/bin/python -m src evaluate prepare
../venv/bin/python -m src evaluate generate
../venv/bin/python -m src evaluate packets
# После внешней слепой оценки пакетов и разбора расхождений:
../venv/bin/python -m src evaluate summarize
```

`prepare` экспортирует фиксированный чекпоинт на CPU и замораживает входы, рубрику, веса и настройки. Ему нужны подготовленная validation, выбранный токенизатор и checkpoint шага 5000; на новой машине перенесите этот checkpoint вместе с артефактами подготовки. `generate --model monolm|qwen` позволяет продолжить только нужную часть. Готовые ответы повторно не генерируются. Инференс использует BF16/SDPA, отдельный суммарный бюджет 30 минут GPU и не запускается одновременно с обучением. Перенос каталогов поддерживается через `MONOLM_ARTIFACTS`.

Оценки выполняются тремя слепыми подагентами вне Python-процесса: пакеты по 10 ответов, 40 повторных оценок и шесть контролей каждому судье. API-ключ не требуется. Сохранённые оценки воспроизводят таблицы отчёта; повторное получение самих оценок требует внешнего оценщика. Продолжения и QA оцениваются раздельно; «минимально приемлемо» — мягкий порог, не утверждение о безошибочной или фактически достоверной речи. Qwen отличается размером и обучающими данными, а оценки подагентов не заменяют носителя языка.

Ноутбук [evaluation.ipynb](notebooks/evaluation.ipynb) сохранён с выполненными ячейками, графиками и примерами — его результаты можно посмотреть прямо в Git. Для повторного вычисления ячеек нужны локальные артефакты `artifacts/evaluation/bg-mini-v1/`, исключённые из Git. Ноутбук читает их и не запускает модели повторно.

Измеренный результат шага 5000:

| Показатель | MonoLM-346M | Qwen3-0.6B-Base |
|---|---:|---:|
| Минимально приемлемое продолжение | 20/60 (33,3%) | 7/60 (11,7%) |
| Продолжение со всеми оценками 2 | 0/60 | 0/60 |
| Полностью правильный QA-ответ | 0/40 | 19/40 (47,5%) |
| Обе формулировки решены правильно | 0/20 | 8/20 (40%) |

Разность минимальной пригодности MonoLM − Qwen: +21,7 п.п., парный 95% bootstrap-интервал [+6,6; +36,7]. Вывод относится к этим входам, параметрам генерации и мягкой рубрике, а не к общему превосходству модели. 18/18 контролей пройдены; согласие о минимальной пригодности — 23/24, о правильности QA — 16/16. Инференс занял 637,1 секунды GPU. Для нулевых долей bootstrap даёт вырожденный [0; 0]; это не означает нулевую неопределённость истинной вероятности успеха.

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
