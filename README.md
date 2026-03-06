## Evaluation Dimensions and Metrics

**Evaluation Dimensions**

| Category | Task Suffix |
|----------|-------------|
| Camera-Object | `cameraqa`, `cameramask` |
| Inter-Object | `qa`, `objmask` |
| Object-Scene | `sceneqa`, `scenemask` |

**Evaluation Metrics**

- **QA Accuracy**: Answer matching accuracy rate
- **Mask J&F Score**: Average of segmentation mask IoU (J) and boundary F-measure (F)



## Model List

**Models Supporting QA + Mask Tasks**

| Model Family | HuggingFace Model ID |
|--------------|---------------------|
| Sa2VA | `ByteDance/Sa2VA-{x}B` |
| Sa2VA-InternVL3 | `ByteDance/Sa2VA-InternVL3-{x}B` |
| Sa2VA-Qwen2_5-VL | `ByteDance/Sa2VA-Qwen2_5-VL-{x}B` |
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



## Project Structure

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



## Quick Start

**1. Environment Setup**

```bash
conda create -n bench python=3.11
conda activate bench

# Install PyTorch (choose according to your CUDA version, see https://pytorch.org)

pip install -r requirements.txt

# Install flash-attn (recommended to use pre-compiled wheels)
# For Linux: https://github.com/Dao-AILab/flash-attention/releases
# For Windows: https://github.com/sdbds/flash-attention-for-windows/releases
```

**2. Clone Repository**

```bash
git clone https://github.com/LilyYang0504/bench.git
cd bench
```

**3. Configuration File**

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

**4. Download Datasets**

```bash
python download_datasets.py
```

**5. Download Models**

Single model download:
```bash
python download_model.py --model "Huggingface/ModelID"
```

Batch download:

```bash
# Edit conf/model_list.txt
python download_model.py --batch
```

**6. Run Evaluation**

```bash
bash start_eval.sh
```



## UniPixel Model Special Instructions

UniPixel requires additional dependencies. See [UniPixel GitHub](https://github.com/PolyU-ChenLab/UniPixel):

```bash
cd bench
mkdir thirdparty
cd thirdparty
git clone https://github.com/PolyU-ChenLab/UniPixel.git
cd UniPixel
pip install -r requirements.txt
```
