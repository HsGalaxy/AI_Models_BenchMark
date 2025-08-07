import os
import logging
from datetime import datetime
from jinja2 import Environment, FileSystemLoader

logger = logging.getLogger(__name__)

def generate_report(results, model_name):
    """
    Generates a Markdown report from the evaluation results using a Jinja2 template.

    Args:
        results (list): A list of result dictionaries from the evaluators.
        model_name (str): The name of the model that was evaluated.
    """
    try:
        # Calculate overall score (simple average)
        total_score = sum(r['score'] for r in results)
        overall_score = total_score / len(results) if results else 0

        # Setup Jinja2 environment
        # The path is relative to where the script is run from, so we construct it carefully.
        template_dir = os.path.join(os.path.dirname(__file__), 'templates')
        env = Environment(loader=FileSystemLoader(template_dir))
        template = env.get_template('report.md.jinja')

        # Data for the template
        template_data = {
            "model_name": model_name,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "overall_score": overall_score,
            "results": results,
        }

        # Render the report
        report_content = template.render(template_data)

        # Save the report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_model_name = model_name.replace('/', '_').replace(':', '_')
        report_filename = f"report_{safe_model_name}_{timestamp}.md"

        # Ensure the results directory exists
        os.makedirs("results", exist_ok=True)
        report_path = os.path.join("results", report_filename)

        with open(report_path, "w", encoding='utf-8') as f:
            f.write(report_content)

        logger.info(f"Report successfully generated at: {report_path}")
        return report_path

    except Exception as e:
        logger.error(f"Failed to generate report: {e}")
        return None
