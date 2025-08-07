import asyncio
import logging
import importlib
from llm_benchmark.utils import load_config
from llm_benchmark.report import generate_report

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_evaluator_class(benchmark_name):
    """Dynamically imports and returns the evaluator class."""
    # This mapping ensures the correct class name is always used.
    class_name_map = {
        "mmlu": "MMLUEvaluator",
        "gsm8k": "GSM8KEvaluator",
        "math": "MATHEvaluator",
        "humaneval": "HumanEvalEvaluator",
    }

    class_name = class_name_map.get(benchmark_name)
    if not class_name:
        logging.error(f"No evaluator class mapping found for benchmark: '{benchmark_name}'")
        return None

    try:
        module_name = f"llm_benchmark.evaluators.{benchmark_name}"
        module = importlib.import_module(module_name)
        return getattr(module, class_name)
    except (ImportError, AttributeError) as e:
        logging.error(f"Could not find or load evaluator for '{benchmark_name}': {e}")
        return None

async def run_model_evaluation(model_config, eval_config):
    """Runs all configured benchmarks for a single model."""
    logging.info(f"--- Starting evaluation for model: {model_config['name']} ---")

    tasks = []
    benchmark_names = eval_config.get('benchmarks', [])

    for bench_name in benchmark_names:
        EvaluatorClass = get_evaluator_class(bench_name)
        if EvaluatorClass:
            # Pass the top-level model config and the entire evaluation config
            evaluator = EvaluatorClass(model_config, eval_config)
            tasks.append(evaluator.run())

    if not tasks:
        logging.warning(f"No valid benchmarks found for model {model_config['name']}. Skipping.")
        return

    # Run all benchmark tasks concurrently for the current model
    results = await asyncio.gather(*tasks)

    # Filter out any None results from failed benchmarks
    results = [r for r in results if r]

    if results:
        generate_report(results, model_config['name'])
    else:
        logging.warning(f"No results were generated for model {model_config['name']}.")

    logging.info(f"--- Finished evaluation for model: {model_config['name']} ---")

async def main():
    """
    Main function to load configuration and orchestrate the benchmark evaluation.
    """
    logging.info("Starting LLM Benchmark System...")

    try:
        config = load_config()
    except FileNotFoundError:
        return # Error is logged in load_config

    models_to_evaluate = config.get('models', [])
    eval_config = config.get('evaluation', {})

    if not models_to_evaluate:
        logging.error("No models found in the configuration file. Exiting.")
        return

    for model_config in models_to_evaluate:
        await run_model_evaluation(model_config, eval_config)

    logging.info("LLM Benchmark System has finished all evaluations.")

if __name__ == "__main__":
    # Note: On Windows, the default event loop policy might cause issues
    # with multiprocessing in some cases. If needed, one could add:
    # if sys.platform == "win32":
    #     asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
