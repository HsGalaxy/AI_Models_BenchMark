import yaml
import asyncio
import logging
from openai import AsyncOpenAI, APIError

# Setup logger
logger = logging.getLogger(__name__)

def load_config(config_path="configs/config.yaml"):
    """
    Loads the YAML configuration file.
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logger.error(f"Configuration file not found at {config_path}. Please create it from the template.")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file: {e}")
        raise

async def call_api(model_config, messages, max_tokens, temperature):
    """
    Makes an asynchronous call to an OpenAI-compatible API with retry logic.

    Args:
        model_config (dict): A dictionary containing model configuration like
                             'api_key', 'api_base', and 'model_name'.
        messages (list): A list of message dictionaries for the chat prompt.
        max_tokens (int): The maximum number of tokens to generate.
        temperature (float): The sampling temperature.

    Returns:
        The model's response content as a string, or None if an error occurs
        after all retries.
    """
    client = AsyncOpenAI(
        api_key=model_config['api_key'],
        base_url=model_config['api_base'],
    )

    max_retries = 5
    backoff_factor = 2
    initial_delay = 1

    for attempt in range(max_retries):
        try:
            response = await client.chat.completions.create(
                model=model_config['model_name'],
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return response.choices[0].message.content
        except APIError as e:
            logger.warning(f"API error occurred (attempt {attempt + 1}/{max_retries}): {e}. Retrying...")
            if attempt + 1 == max_retries:
                logger.error("Max retries reached. API call failed.")
                return None
            delay = initial_delay * (backoff_factor ** attempt)
            await asyncio.sleep(delay)
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}")
            return None
    return None
