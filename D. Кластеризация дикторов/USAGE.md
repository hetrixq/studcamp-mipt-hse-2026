### Запуск и структура проекта

Этот проект делает кластеризацию дикторов по `.wav` файлам и пишет `submission.csv` .

### Быстрый старт

Из папки `D. Кластеризация дикторов/` :

```bash
python3 -m speaker_clustering --data_dir audio --out submission.csv --clusters 50
```

То же самое, но из корня репозитория:

```bash
python3 -m speaker_clustering --data_dir "D. Кластеризация дикторов/audio" --out submission.csv --clusters 50
```

### Установка окружения

Команды из корня репозитория:

```bash
# 0) выйти из окружения (если активировано)
deactivate 2>/dev/null || true

# 1) снести старое окружение
rm -rf .venv

# 2) создать новое на python 3.14
python3.14 -m venv .venv
source .venv/bin/activate

# 3) обновить pip
pip install -U pip

# 4) поставить torch/torchaudio (подберутся доступные под py3.14)
pip install torch torchaudio

# 5) нужное для кластеризации + прогрессбар
pip install numpy pandas scikit-learn tqdm

# 6) критично: speechbrain из github (фикс совместимости с torchaudio 2.9+)
pip install "git+https://github.com/speechbrain/speechbrain.git"

# 7) на всякий случай: декодер wav 
pip install soundfile

pip install torchcodec
```

Зависимости также перечислены в `requirements.txt` , но для `speechbrain` используется установка из GitHub.

### Что делает пайплайн

* (опционально) трим тишины (энергетический, без дополнительных зависимостей)
* эмбеддинги диктора: SpeechBrain ECAPA-TDNN (VoxCeleb)
* L2-нормализация эмбеддингов (для cosine-геометрии)
* Agglomerative clustering (cosine, average) в фиксированное число кластеров `--clusters`
* запись `submission.csv` с колонками `filename,speaker_id`

### CLI параметры

Справка:

```bash
python3 -m speaker_clustering --help
```

Ключевые флаги:

* `--data_dir`: папка с `.wav`
* `--out`: куда писать CSV
* `--clusters`: число кластеров (по условию обычно `50`)
* `--trim_silence 1`,  `--trim_db`,  `--min_speech_ms`: трим тишины
* `--chunk_sec`,  `--hop_sec`: усреднение эмбеддингов по чанкам (для длинных записей)
* `--batch_size`: батчинг (обычно 8–16 для MPS)
* `--device auto|cpu|cuda|mps`: выбор устройства
* `--verbose 1`: показать шумные логи зависимостей (HF/SpeechBrain и т.п.)
* `--warnings 1`: показать warnings (по умолчанию скрыты)

### Мини-описание модулей

* `speaker_clustering/cli.py`: парсинг аргументов, сбор `RunConfig`
* `speaker_clustering/config.py`: dataclass конфиг запуска
* `speaker_clustering/pipeline.py`: основной пайплайн (чтение wav → эмбеддинги → кластеризация → CSV)
* `speaker_clustering/audio.py`: `to_mono`,  `resample_if_needed`,  `trim_silence_energy`
* `speaker_clustering/embedding.py`: загрузка SpeechBrain модели + `chunk_and_average_embedding`
* `speaker_clustering/clustering.py`: agglomerative clustering (cosine/average) с совместимостью разных sklearn
* `speaker_clustering/device.py`: выбор `cpu/cuda/mps/auto`
* `speaker_clustering/seed.py`: фиксация seed (numpy/torch/random)
* `speaker_clustering/fs.py`: поиск и сортировка `.wav`
* `speaker_clustering/speechbrain_compat.py`: патч совместимости SpeechBrain на macOS MPS

### Нюансы / частые проблемы

* Первый запуск SpeechBrain скачивает веса модели (интернет нужен хотя бы один раз).
* На macOS полезны `soundfile`/`torchcodec`, если `torchaudio` спотыкается на декодировании WAV.
