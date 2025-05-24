# ULTRABENCH: Benchmarking LLMs under Extreme Fine-grained Text Generation

This repository provides the data and scripts for evaluating large language models under zero-shot prompting as part of the ULTRABENCH benchmark.

---

## 📦 Dataset

The dataset is publicly available on Hugging Face:  
👉 [ylf1017/ultrabench](https://huggingface.co/datasets/ylf1017/ultrabench)

After downloading, place the files in the `data` directory:

```
data/
└── ultrabench_train.jsonl
└── ultrabench_test.jsonl
```

---

## 🔧 Setup Instructions

### 1. Configure API Key

Create a `.env` file in the root directory and add your OpenAI API key:

```env
OPENAI_API_KEY=<your_openai_api_key>
```

### 2. Install Dependencies
We recommend using uv for fast and reliable dependency installation:

```bash
pip install --upgrade pip
pip install uv
uv pip install "sglang[all]>=0.4.6.post4"
uv pip install -r requirements.txt
```

### 3. Run Evaluation
Use the provided script to launch evaluation:

```bash
bash zero_shot.sh
```

## 📄 License
This dataset and code are released under the CC BY 4.0 License.
You are free to use, modify, and distribute the materials with proper attribution.


## 🔗 Citation
If you use this dataset or benchmark in your research, please cite:

```
@article{yun2025ultragen,
  title={UltraGen: Extremely Fine-grained Controllable Generation via Attribute Reconstruction and Global Preference Optimization},
  author={Yun, Longfei and Peng, Letian and Shang, Jingbo},
  journal={arXiv preprint arXiv:2502.12375},
  year={2025}
}
```
