# FastSpeech 2 TTS: English (Svarah) and Telugu (IndicTTS)

Two separate FastSpeech 2 models, based on https://github.com/ming024/FastSpeech2, with HiFi-GAN as the vocoder.

## Structure
`English/` and `Telugu/` each hold the code, `config.yaml`, `config/<lang>/*.yaml` (used by the scripts) and `checkpoints/`.
`Results/` holds loss plots, generated test audio and variance-adaptor plots. `Report/Technical_Report.pdf` is the report.

## Setup
`pip install -r requirements.txt` (Python 3.13 was used for the final run).

## Commands (run inside `Telugu/` or `English/`)
TODO: edit these to match exactly what you ran.
    python prepare_align.py config/Telugu/preprocess.yaml
    python preprocess.py config/Telugu/preprocess.yaml
    python train.py -p config/Telugu/preprocess.yaml -m config/Telugu/model.yaml -t config/Telugu/train.yaml
    python inference.py --lang Telugu --step 25000 --out ../Results

## Checkpoints
Stored with Git LFS in `<lang>/checkpoints/`. Model weights only (no optimizer state).

## Telugu changes
TODO: symbol set (Telugu characters as "@<char>" tokens, "@sp" between words), alignment, text handling in inference.py.

## Text normalization at inference
Digits are not in the Telugu symbol set, so the numbers in T4 and E4 are written out as words. Originals are in `Results/<lang>/summary.json`.
