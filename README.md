# DNAIC

<div align='center'>
  <h2>
    Distilled Non-Autoregressive Image Captioning
  </h2>
</div>

<div align='center'>
  <img src='https://img.shields.io/github/license/yuanzhu/DNAIC?label=License&style=flat&color=7e57c2' alt='License'>
  <img src='https://img.shields.io/github/contributors/yuanzhu/DNAIC?label=Contributors&style=flat&color=1e88e5' alt='Contributors'>
</div>

## Pipeline

<div align='center'>
    <img src='pipeline.svg' width=100%>
</div>

## Data Preparation

- The dataset should be placed in the `mscoco/` directory

```bash
mscoco/
|--feature/
  |--coco2014/
    |--train2014/
    |--value2014/
    |--test2014/
|--misc/
|--sent/
|--txt/
```

Download link: [mscoco](https://pan.baidu.com/s/1KSYMOxvs-W6brrgpsZI9Aw?pwd=rain)

## Pre-trained Models Preparation

- Pre-trained models should be placed in the `pretrained_models/` directory

```bash
pretrained_models/
|-- mplug/
|-- swin-large-patch4-window12-384/
|-- vit/
```

Download link: [pretrained_models](https://pan.baidu.com/s/19h-6fc48hiyFZupPSf_vuQ?pwd=rain)

## Training Process

The model can be trained on NVIDIA GPU with CUDA support or Apple Silicon with MPS support (CPUs are not recommended).

1. Environment Setup

   Python 3.12

   ```bash
   pip install -r requirements.txt
   ```

2. Training Command

   ```bash
   bash experiments/train.sh
   ```

## Evaluation

1. Evaluation Script

   ```bash
   bash experiments/eval.sh
   ```

2. Evaluation Metrics
   - Image Captioning: BLEU, METEOR, ROUGE-L, CIDEr
   - Speed: Latency, SpeedUp

## Inference

1. Inference Performance

   ```bash
   bash experiments/inference.sh
   ```

2. WebUI

   A Gradio-based web UI is also provided for users to perform inference through a graphical interface.

   ```py
   python webui.py
   ```

## Credits

This project is based on the following resources:

- MSCOCO
- mPLUG
- EENAIC

## License

This project is licensed under the terms specified in [LICENSE](LICENSE).
