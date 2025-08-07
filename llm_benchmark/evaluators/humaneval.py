import multiprocessing
import random
from datasets import load_dataset
from ..benchmark import BenchmarkEvaluator

# It's crucial to run untrusted code in a separate process.
# This function will be the target for the multiprocessing.Process.
def execute_code(code_to_run, result_queue):
    try:
        # Using a restricted globals dictionary for a little extra safety
        exec(code_to_run, {})
        result_queue.put(True)
    except Exception:
        result_queue.put(False)

class HumanEvalEvaluator(BenchmarkEvaluator):
    """
    Evaluator for the HumanEval benchmark.
    This benchmark tests the ability of a model to generate functional code.
    """

    @property
    def benchmark_name(self):
        return "humaneval"

    def load_data(self):
        """
        Loads the HumanEval dataset.
        """
        # For HumanEval, we evaluate on the entire test set as is standard.
        # It's small enough (164 problems).
        dataset = load_dataset("openai_humaneval", split="test")
        # Using the full test set as is standard for HumanEval.
        return list(dataset)


    def format_prompt(self, sample):
        """
        The prompt in HumanEval is already well-formed.
        We just need to present it to the model.
        """
        return [{"role": "user", "content": sample['prompt']}]

    def process_response(self, response, sample):
        """
        Processes the model's response by executing it against the test cases.
        """
        # The model should generate the body of the function.
        # We combine the prompt (which includes the function signature) with the response.
        full_code = sample['prompt'] + response

        # We also need to append the test cases to check for correctness.
        # The entry_point tells us the function name to use in the tests.
        test_code = sample['test'] + f"\n\ncheck({sample['entry_point']})"

        code_to_run = full_code + "\n\n" + test_code

        result_queue = multiprocessing.Queue()
        process = multiprocessing.Process(target=execute_code, args=(code_to_run, result_queue))

        process.start()
        # A timeout is critical to handle code that might hang.
        process.join(timeout=10.0)

        if process.is_alive():
            process.terminate()
            process.join()
            is_correct = False
            error = "Timeout"
        else:
            if process.exitcode != 0:
                is_correct = False
                error = f"Process exited with code {process.exitcode}"
            else:
                is_correct = result_queue.get()
                error = "Failed Execution" if not is_correct else None

        return {
            "correct": is_correct,
            "model_answer_full": response,
            "error": error
        }

    def aggregate_results(self, results):
        """
        Calculates the pass@1 score.
        """
        passed_count = sum(1 for r in results if r.get("correct", False))
        total_count = len(results)
        return passed_count / total_count if total_count > 0 else 0.0
