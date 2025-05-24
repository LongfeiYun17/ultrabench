# ULTRABENCH: Benchmarking LLMs under Extreme Fine-grained Text Generation

This script evaluates large language models under the **zero-shot** prompting baseline as part of the ULTRABENCH benchmark.

## 🔧 Setup Instructions

### 1. Configure API Key  
Create a `.env` file in the root directory and add your OpenAI API key:

```env
OPENAI_API_KEY=<your_openai_api_key>
```

### 2. Install Dependencies  
We recommend using [uv](https://github.com/astral-sh/uv) for faster and more reliable dependency installation.

```bash
pip install --upgrade pip
pip install uv
uv pip install "sglang[all]>=0.4.6.post4"
uv pip install -r requirements.txt
```

### 3. Run the Evaluation Script

```bash
bash zero_shot.sh
```