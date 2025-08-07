import random
from datasets import load_dataset
from ..benchmark import BenchmarkEvaluator

class MMLUEvaluator(BenchmarkEvaluator):
    """
    Evaluator for the MMLU (Massive Multitask Language Understanding) benchmark.
    """

    @property
    def benchmark_name(self):
        return "mmlu"

    def load_data(self):
        """
        Loads the MMLU dataset. For demonstration, we'll use a small subset.
        In a real scenario, you might want to use the full 'all' split.
        """
        # Using the 'all' configuration and then slicing for manageability
        # The 'auxiliary_train' split is smaller and good for creating few-shot examples.
        self.few_shot_data = list(load_dataset("cais/mmlu", "all", split="auxiliary_train"))

        # Using the 'test' split for actual evaluation
        dataset = load_dataset("cais/mmlu", "all", split="test")
        # Using 200 samples for a more comprehensive but still manageable test run.
        return random.sample(list(dataset), 200)

    def format_prompt(self, sample):
        """
        Formats a few-shot prompt for the MMLU task.
        """
        k_shot = self.benchmark_config.get(self.benchmark_name, {}).get("k_shot", 5)

        # Get k-shot examples from the auxiliary train set
        few_shot_examples = random.sample(self.few_shot_data, k_shot)

        prompt = "The following are multiple choice questions (with answers).\n\n"

        for ex in few_shot_examples:
            prompt += self._format_single_question(ex, with_answer=True)

        prompt += self._format_single_question(sample, with_answer=False)

        return [{"role": "user", "content": prompt}]

    def _format_single_question(self, sample, with_answer=True):
        question = f"{sample['question']}\n"
        choices = sample['choices']
        for i, choice in enumerate(choices):
            question += f"{chr(65 + i)}. {choice}\n"
        question += "Answer:"
        if with_answer:
            question += f" {chr(65 + sample['answer'])}\n\n"
        return question

    def process_response(self, response, sample):
        """
        Processes the model's response to check for correctness.
        The model should output a single letter (A, B, C, D).
        """
        # Clean up the response
        parsed_response = response.strip().upper()

        # Get the first letter if the model is verbose
        if parsed_response:
            parsed_response = parsed_response[0]
        else:
            return {"correct": False, "parsed_answer": "N/A"}

        correct_answer_char = chr(65 + sample['answer'])

        is_correct = parsed_response == correct_answer_char

        return {
            "correct": is_correct,
            "model_answer": response,
            "parsed_answer": parsed_response,
            "correct_answer": correct_answer_char
        }
