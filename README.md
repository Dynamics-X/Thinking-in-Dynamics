<div align="center" style="font-family: charter;">

<h1>
    <img src="docs/resources/icon.png" width="80" style="position: relative; top: 15px; left: 10px;"/>
    <i>Thinking in Dynamics</i>:<br/> How Multimodal Large Language Models Perceive, Track, and Reason Dynamics in Physical 4D World
</h1>

<img src="docs/resources/paper.png" width="50%"/>
<br />

<a href="https://arxiv.org/abs/2512.03000" target="_blank">
    <img alt="arXiv" src="https://img.shields.io/badge/arXiv-2512.03000-red?logo=arxiv" height="20" />
</a>
<a href="https://dxuan77.github.io/Thinking-in-Dynamics/" target="_blank">
    <img alt="Website" src="https://img.shields.io/badge/🌎_Website-thinking--in--dynamics-blue.svg" height="20" />
</a>
<a href="https://huggingface.co/datasets/kairunwen/DynamicVerse" target="_blank">
    <img alt="HF Dataset: DynamicVerse" src="https://img.shields.io/badge/%F0%9F%A4%97%20_Dataset-DynamicVerse-ffc107?color=ffc107&logoColor=white" height="20" />
</a>
<a href="https://github.com/Dynamics-X/DynamicVerse" target="_blank">
    <img alt="GitHub" src="https://img.shields.io/badge/GitHub-DynamicVerse-black?logo=github" height="20" />
</a>
<a href="https://www.youtube.com/watch?v=0h7XysIpG8Y" target="_blank">
    <img alt="Demo Video" src="https://img.shields.io/badge/▶_Demo-YouTube-red?logo=youtube" height="20" />
</a>

<div>
    Kairun Wen<sup>*</sup>,&ensp;
    Yuzhi Huang<sup>*</sup>,&ensp;
    Runyu Chen<sup>*</sup>,&ensp;
    Hui Zheng,&ensp;
    Yunlong Lin,&ensp;
    Panwang Pan,&ensp;
    Chenxin Li,&ensp;
    Wenyan Cong,&ensp;
    Jian Zhang,&ensp;
    Junbin Lu,&ensp;
    Chenguo Lin,&ensp;
    Dilin Wang,&ensp;
    Zhicheng Yan,&ensp;
    Hongyu Xu,&ensp;
    Justin Theiss,&ensp;
    Yue Huang,&ensp;
    Xinghao Ding,&ensp;
    Rakesh Ranjan,&ensp;
    Zhiwen Fan
</div>

<img src="docs/resources/teaser.png" width="100%"/>
<p align="justify"><i>Humans inhabit a physical 4D world where geometric structure and semantic content evolve over time. While current Multimodal Large Language Models (MLLMs) excel in static visual understanding, can they also be adept at "thinking in dynamics" — perceiving, tracking and reasoning about spatio-temporal dynamics in evolving scenes? We introduce <strong>Dyn-Bench</strong>, a large-scale benchmark comprising 1k videos, 7k VQA pairs, and 3k dynamic object grounding pairs, systematically assessing MLLMs' ability to perceive, track, and reason about object motion, scene evolution, and camera motion in the physical 4D world.</i></p>

</div>

## Release

- `2025-XX-XX` 🚀 Dyn-Bench evaluation code and benchmark released.

## Contents

- [Release](#release)
- [Contents](#contents)
- [Dyn-Bench](#dyn-bench)
- [Results](#results)
- [Run Your Own Evaluation](#run-your-own-evaluation)
  - [Benchmark](#benchmark)
  - [Installation](#installation)
  - [Configuration](#configuration)
  - [Evaluation](#evaluation)
- [Model List](#model-list)
- [UniPixel Special Instructions](#unipixel-special-instructions)
- [Acknowledgement](#acknowledgement)
- [Citation](#citation)

## Dyn-Bench

**Overview:** We introduce **Dyn-Bench**, a spatio-temporal dynamics reasoning benchmark built from diverse real-world and synthetic video datasets. Dyn-Bench comprises **1k videos**, **7k visual question answering (VQA) pairs**, and **3k dynamic object grounding pairs**, enabling robust and scalable evaluation of spatio-temporal understanding across three evaluation dimensions.

**Evaluation Dimensions**

| Category | Task Suffix | Description |
|----------|-------------|-------------|
| Camera-Object | `cameraqa`, `cameramask` | Reasoning about object dynamics relative to camera motion |
| Inter-Object | `qa`, `objmask` | Reasoning about interactions and relative dynamics between objects |
| Object-Scene | `sceneqa`, `scenemask` | Reasoning about how objects interact with and evolve within the scene |

**Evaluation Metrics**

- **QA Accuracy**: Answer matching accuracy rate for VQA tasks
- **Mask J&F Score**: Average of segmentation mask IoU (J) and boundary F-measure (F) for grounding tasks

## Results

**Evaluation Setup:** We probe general, spatial, and region-level MLLMs across all three evaluation dimensions, assessing both linguistic (VQA) and visual (mask grounding) dynamics understanding. We find that existing models cannot simultaneously maintain strong performance in both spatio-temporal reasoning and dynamic object grounding, often producing inconsistent interpretations of motion and interaction.

> Note: Conventional prompting strategies (e.g., chain-of-thought or caption-based hints) provide limited improvement, whereas structured integration approaches — including **Mask-Guided Fusion** and **Spatio-Temporal Textual Cognitive Map (ST-TCM)** — significantly enhance MLLMs' dynamics perception and spatio-temporal reasoning in the physical 4D world.

## Run Your Own Evaluation

### Benchmark

Our benchmark is hosted on [HuggingFace](https://huggingface.co/datasets/kairunwen/DynamicVerse). You can access the benchmark data using:

```python
# NOTE: pip install datasets

from datasets import load_dataset
dyn_bench = load_dataset("kairunwen/DynamicVerse")
print(dyn_bench)
```

### Installation

**1. Environment Setup**

```bash
conda create -n bench python=3.11
conda activate bench

# Install PyTorch (choose according to your CUDA version, see https://pytorch.org)

pip install -r requirements.txt

# Install flash-attn (recommended to use pre-compiled wheels)
# For Linux:   https://github.com/Dao-AILab/flash-attention/releases
# For Windows: https://github.com/sdbds/flash-attention-for-windows/releases
```

**2. Clone Repository**

```bash
git clone https://github.com/LilyYang0504/bench.git
cd bench
```

### Configuration

Edit `conf/config.yaml`:

```yaml
datasets:
  repo_name: "Huggingface/DatasetsRepo"
  datasets_path: "path/for/your/datasets/download"

model:
  model_path: "path/to/model"
  model_name: "Huggingface/ModelID"
  download_path: "path/for/your/model/download"
  device: "cuda"
  torch_dtype: "bfloat16"
  use_flash_attn: true
  trust_remote_code: true

task: "all"  # choices: all / qa / mask

evaluation:
  boundary_threshold: 2

result_path: "results"
```

### Evaluation

**Download datasets:**
```bash
python download_datasets.py
```

**Download a single model:**
```bash
python download_model.py --model "Huggingface/ModelID"
```

**Batch model download** (edit `conf/model_list.txt` first):
```bash
python download_model.py --batch
```

**Run evaluation:**
```bash
bash start_eval.sh
```

### Project Structure

```
bench/
├── conf/
│   ├── config.yaml
│   └── model_list.txt
├── utils/
├── thirdparty/
├── download_datasets.py
├── download_model.py
├── eval.py
└── start_eval.sh
```

## Model List

**Models Supporting QA + Mask Tasks**

| Model Family | HuggingFace Model ID |
|--------------|---------------------|
| Sa2VA | `ByteDance/Sa2VA-{x}B` |
| Sa2VA-InternVL3 | `ByteDance/Sa2VA-InternVL3-{x}B` |
| Sa2VA-Qwen2.5-VL | `ByteDance/Sa2VA-Qwen2_5-VL-{x}B` |
| Sa2VA-Qwen3-VL | `ByteDance/Sa2VA-Qwen3-VL-{x}B` |
| UniPixel | `PolyU-ChenLab/UniPixel-{x}B` (requires additional installation) |

**Models Supporting QA Tasks Only**

| Model Family | HuggingFace Model ID |
|--------------|---------------------|
| InternVL3 | `OpenGVLab/InternVL3-{x}B` |
| InternVL3.5 | `OpenGVLab/InternVL3_5-{x}B` |
| Qwen2.5-VL | `Qwen/Qwen2.5-VL-{x}B-Instruct` |
| Qwen3-VL | `Qwen/Qwen3-VL-{x}B-Instruct` |
| Qwen3-VL-MoE | `Qwen/Qwen3-VL-235B-A22B-Instruct` |
| LLaVA-OneVision | `lmms-lab/LLaVA-One-Vision-1.5-{x}B-Instruct` |
| SpaceR-SFT | `RUBBISHLIKE/SpaceR-SFT-{x}B` |
| VST | `rayruiyang/VST-{x}B-RL` |
| Spatial-SSRL | `internlm/Spatial-SSRL-{x}B` |
| SpatialLadder | `hongxingli/SpatialLadder-{x}B` |

> Replace `{x}B` with the actual model parameter size. Please check HuggingFace for available sizes.

## UniPixel Special Instructions

UniPixel requires additional dependencies. See [UniPixel GitHub](https://github.com/PolyU-ChenLab/UniPixel):

```bash
cd bench
mkdir thirdparty
cd thirdparty
git clone https://github.com/PolyU-ChenLab/UniPixel.git
cd UniPixel
pip install -r requirements.txt
```

## Acknowledgement

We gratefully acknowledge the open-source community and the authors of the video datasets and foundation models used in constructing Dyn-Bench. Our evaluation framework builds upon the excellent toolkits provided by the community for evaluating multimodal large language models.

## Citation

If you find our paper and code useful in your research, please consider giving us a star :star: and citing our work :pencil: :)

```bibtex
@misc{wen2025dynamicverse,
    title={DynamicVerse: A Physically-Aware Multimodal Framework for 4D World Modeling},
    author={Kairun Wen and Yuzhi Huang and Runyu Chen and Hui Zheng and Yunlong Lin and Panwang Pan and Chenxin Li and Wenyan Cong and Jian Zhang and Junbin Lu and Chenguo Lin and Dilin Wang and Zhicheng Yan and Hongyu Xu and Justin Theiss and Yue Huang and Xinghao Ding and Rakesh Ranjan and Zhiwen Fan},
    year={2025},
    eprint={2512.03000},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```
