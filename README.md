# scDecAge

Implementation of **scDecAge**, a framework for individual-level chronological-age prediction from single-cell transcriptomes.



## Repository layout

```text
scDecAge/
|-- scdecage/                # Installable Python package
|-- scripts/                 # Preprocessing, training, prediction, validation
|-- pyproject.toml
`-- requirements.txt
```

## Installation

```bash
git clone git@github.com:yjs193/scDecAge.git
cd scDecAge
conda create -n scdecage python=3.11 -y
conda activate scdecage
pip install -e .
```


FlashAttention is optional. If it is unavailable, the cell encoder uses
PyTorch scaled dot-product attention with the same learned parameters.


## Training

```bash
python scripts/train_scdecage.py \
  --data-root /path/to/scDecAge_Data \
```

## Prediction

```bash
python scripts/predict_scdecage.py \
  --checkpoint xx.pth \
  --data-root /path/to/scDecAge_Data \
  --dataset-dir /path/to/scDecAge_Data/datasets/OneK1K_CELLxGENE \
  --split test \
```



## License

MIT License. See [LICENSE](LICENSE).
