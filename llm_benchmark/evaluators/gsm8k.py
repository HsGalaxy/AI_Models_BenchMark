import re
import random
from datasets import load_dataset
from ..benchmark import BenchmarkEvaluator

class GSM8KEvaluator(BenchmarkEvaluator):
    """
    Evaluator for the GSM8K (Grade School Math 8K) benchmark.
    This benchmark tests the mathematical reasoning capabilities of a model.
    It uses chain-of-thought prompting for best results.
    """

    @property
    def benchmark_name(self):
        return "gsm8k"

    def load_data(self):
        """
        Loads the GSM8K dataset.
        """
        dataset = load_dataset("gsm8k", "main")
        self.few_shot_data = list(dataset['train'])
        # Using the full test set as is standard for GSM8K.
        return list(dataset['test'])

    def format_prompt(self, sample):
        """
        Formats a few-shot, chain-of-thought prompt for the GSM8K task.
        """
        k_shot = self.benchmark_config.get(self.benchmark_name, {}).get("k_shot", 8)

        few_shot_examples = random.sample(self.few_shot_data, k_shot)

        prompt = "The following are grade school math questions. Please solve them step-by-step.\n\n"

        for ex in few_shot_examples:
            prompt += f"Question: {ex['question']}\n"
            prompt += f"Answer:\n{ex['answer']}\n\n"

        prompt += f"Question: {sample['question']}\n"
        prompt += "Answer:\n"

        return [{"role": "user", "content": prompt}]

    def process_response(self, response, sample):
        """
        Processes the model's response to extract the final numerical answer.
        The ground truth answer is in the format "#### <number>".
        We need to extract the number from both the model's response and the ground truth.
        """
        # Extract ground truth answer
        try:
            true_answer_str = sample['answer'].split('####')[-1].strip()
            true_answer = float(true_answer_str.replace(',', ''))
        except (ValueError, IndexError):
            return {"correct": False, "error": "Could not parse ground truth answer."}

        # Extract model's final answer
        model_answer_str = ""
        # The model might use the same "####" format or just output the number.
        # We look for the last number in the response.
        # A simple regex to find numbers (including decimals and commas)
        matches = re.findall(r'[\d,.]+', response)
        if matches:
            # Take the last number found as the final answer
            model_answer_str = matches[-1]
            try:
                # Remove commas for safe conversion
                model_answer = float(model_answer_str.replace(',', ''))
            except ValueError:
                return {"correct": False, "parsed_answer": model_answer_str}
        else:
            return {"correct": False, "parsed_answer": "N/A"}

        is_correct = abs(model_answer - true_answer) < 1e-6 # Compare floats

        return {
            "correct": is_correct,
            "model_answer_full": response,
            "parsed_answer": model_answer,
            "correct_answer": true_answer
        }
