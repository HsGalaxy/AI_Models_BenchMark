# AI Model Benchmark System

This project is a comprehensive, automated, and elegant AI model benchmark system designed to evaluate the performance of large language models (LLMs). It supports any OpenAI-compatible API endpoint.

## Features

- **Comprehensive Evaluation**: Includes benchmarks for a wide range of capabilities:
  - **MMLU**: General knowledge and problem-solving.
  - **GSM8K**: Grade-school math and reasoning (using Chain-of-Thought).
  - **MATH**: Advanced competitive mathematics.
  - **HumanEval**: Code generation, with safe execution of generated code.
- **Fully Automated**: A single command runs the entire suite for all configured models.
- **Objective Scoring**: Provides scores for each benchmark and a final, aggregated score.
- **Elegant & Beautiful Results**: Generates clean, readable Markdown reports for each model evaluation.
- **Extensible**: Easily add new benchmarks by creating a new evaluator module.
- **Fast & Efficient**: Uses `asyncio` to run evaluations in parallel, significantly speeding up the process.

## Project Structure

```
.
├── configs/
│   ├── config.yaml         # Your local configuration file
│   └── config.yaml.template # A template to get you started
├── llm_benchmark/
│   ├── evaluators/
│   │   ├── mmlu.py
│   │   ├── gsm8k.py
│   │   ├── math.py
│   │   └── humaneval.py
│   ├── templates/
│   │   └── report.md.jinja   # Jinja2 template for the report
│   ├── __init__.py
│   ├── benchmark.py        # Abstract base class for evaluators
│   ├── report.py           # Report generation logic
│   └── utils.py            # API client and config loader
├── results/                  # Output reports are saved here
├── main.py                   # Main execution script
├── requirements.txt          # Project dependencies
└── README.md                 # This file
```

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

2.  **Install dependencies:**
    It is recommended to use a virtual environment.
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```

## Configuration

1.  **Create your configuration file:**
    Copy the template to create your own config file.
    ```bash
    cp configs/config.yaml.template configs/config.yaml
    ```

2.  **Edit `configs/config.yaml`:**
    Open the `config.yaml` file and enter the details for the model(s) you want to evaluate.
    - `name`: A unique, friendly name for your model configuration.
    - `api_key`: Your API key.
    - `api_base`: The base URL for the API endpoint (e.g., `https://api.openai.com/v1`).
    - `model_name`: The specific model identifier to be used in API calls (e.g., `gpt-4`).

    You can also select which benchmarks to run by editing the `benchmarks` list.

## How to Run

Once your `configs/config.yaml` is set up, run the benchmark suite with a single command:

```bash
python main.py
```

The script will start the evaluation process, showing progress bars for each benchmark. Upon completion for a given model, a detailed Markdown report will be saved in the `results/` directory.

The script may take a significant amount of time to run, depending on the number of models, the benchmarks selected, and the API response times.
