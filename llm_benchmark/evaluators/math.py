import re
import random
from datasets import load_dataset
from ..benchmark import BenchmarkEvaluator

class MATHEvaluator(BenchmarkEvaluator):
    """
    Evaluator for the MATH (Mathematical Problem Solving) benchmark.
    This is a challenging benchmark requiring deep mathematical reasoning.
    """

    @property
    def benchmark_name(self):
        return "math"

    def load_data(self):
        """
        Loads the MATH dataset from 'nlile/hendrycks-MATH-benchmark'.
        This version only has a 'train' split, so we create our own test set
        by splitting the data.
        """
        dataset = load_dataset("nlile/hendrycks-MATH-benchmark", split="train")

        # Shuffle the dataset
        shuffled_dataset = dataset.shuffle(seed=42)

        # We'll use a portion for testing and the rest for few-shot examples.
        # Let's use 500 samples for the test set.
        test_size = 500

        self.few_shot_data = list(shuffled_dataset.select(range(test_size, len(shuffled_dataset))))
        test_dataset = shuffled_dataset.select(range(test_size))

        return list(test_dataset)

    def format_prompt(self, sample):
        """
        Formats a few-shot, chain-of-thought prompt for the MATH task.
        """
        k_shot = self.benchmark_config.get(self.benchmark_name, {}).get("k_shot", 4)

        few_shot_examples = random.sample(self.few_shot_data, k_shot)

        prompt = "The following are challenging math problems. Please solve them step-by-step, and put the final answer in a box like \\boxed{answer}.\n\n"

        for ex in few_shot_examples:
            prompt += f"Problem: {ex['problem']}\n"
            prompt += f"Solution:\n{ex['solution']}\n\n"

        prompt += f"Problem: {sample['problem']}\n"
        prompt += "Solution:\n"

        return [{"role": "user", "content": prompt}]

    def process_response(self, response, sample):
        """
        Processes the model's response to extract the final answer from a \\boxed{} block.
        """
        # Extract ground truth answer
        true_answer = self.extract_boxed_answer(sample['solution'])
        if true_answer is None:
            return {"correct": False, "error": "Could not parse ground truth answer."}

        # Extract model's answer
        model_answer = self.extract_boxed_answer(response)
        if model_answer is None:
            return {"correct": False, "parsed_answer": "N/A"}

        # For now, we do a simple string comparison after some normalization.
        # A more advanced system would use a symbolic math library.
        is_correct = self.is_equiv(model_answer, true_answer)

        return {
            "correct": is_correct,
            "model_answer_full": response,
            "parsed_answer": model_answer,
            "correct_answer": true_answer
        }

    def extract_boxed_answer(self, text):
        """Extracts the content from the last \\boxed{} block in the text."""
        matches = re.findall(r'\\boxed\{(.+?)\}', text)
        if matches:
            return matches[-1]
        return None

    def is_equiv(self, str1, str2):
        """
        A simplified check for equivalence between two answers.
        This removes spaces and commas. A more robust solution is much more complex.
        """
        def normalize(s):
            return s.replace(" ", "").replace(",", "")

        return normalize(str1) == normalize(str2)
