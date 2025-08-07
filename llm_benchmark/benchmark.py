import logging
from abc import ABC, abstractmethod
from tqdm.asyncio import tqdm
from datasets import load_dataset
from .utils import call_api

logger = logging.getLogger(__name__)

class BenchmarkEvaluator(ABC):
    """
    Abstract base class for a benchmark evaluator.

    This class provides the core structure for running an evaluation. Subclasses
    must implement the abstract methods to provide benchmark-specific logic.
    """

    def __init__(self, model_config, benchmark_config):
        self.model_config = model_config
        self.benchmark_config = benchmark_config
        self.api_params = self.benchmark_config.get('api_params', {})

    @property
    @abstractmethod
    def benchmark_name(self):
        """A string identifier for the benchmark."""
        pass

    @abstractmethod
    def load_data(self):
        """Loads and returns the dataset for the benchmark."""
        pass

    @abstractmethod
    def format_prompt(self, sample):
        """Formats the prompt for a given sample from the dataset."""
        pass

    @abstractmethod
    def process_response(self, response, sample):
        """Processes the model's response and returns the evaluation result."""
        pass

    async def run(self):
        """
        Runs the full evaluation for this benchmark.
        """
        logger.info(f"Running benchmark: {self.benchmark_name} for model: {self.model_config['name']}")
        dataset = self.load_data()

        results = []

        # Using tqdm.asyncio for an async-compatible progress bar
        for sample in tqdm(dataset, desc=f"Evaluating {self.benchmark_name}"):
            prompt_messages = self.format_prompt(sample)

            response = await call_api(
                model_config=self.model_config,
                messages=prompt_messages,
                max_tokens=self.api_params.get('max_tokens', 1024),
                temperature=self.api_params.get('temperature', 0.1)
            )

            if response is None:
                # Handle API call failure
                result = {"correct": False, "error": "API call failed"}
            else:
                result = self.process_response(response, sample)

            results.append(result)

        # Aggregate results
        score = self.aggregate_results(results)
        logger.info(f"Finished benchmark: {self.benchmark_name}. Score: {score:.4f}")

        return {
            "benchmark": self.benchmark_name,
            "model": self.model_config['name'],
            "score": score,
            "total_samples": len(results),
        }

    def aggregate_results(self, results):
        """
        Aggregates individual sample results into a final score.
        By default, calculates accuracy. Can be overridden by subclasses.
        """
        correct_count = sum(1 for r in results if r.get("correct", False))
        total_count = len(results)
        return correct_count / total_count if total_count > 0 else 0.0
