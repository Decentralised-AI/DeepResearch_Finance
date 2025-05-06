import sys
import os

# Get the parent directory path
#parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))

# Add parent directory to sys.path if not already present
#if parent_dir not in sys.path:
#    sys.path.insert(0, parent_dir)



from deepsearcher.configuration import Configuration, init_config
from deepsearcher.online_query import query


config = Configuration()
openai_api_key = "Add the LLM API Key"
config.set_provider_config("llm", "OpenAI", {"model": "o1-mini", "api_key": openai_api_key})
config.set_provider_config("embedding", "OpenAIEmbedding", {"model": "text-embedding-ada-002", "api_key": openai_api_key})

print(config.provide_settings)

init_config(config=config)

from deepsearcher.offline_loader import load_from_local_files

local_path = "/Users/petros-pavlosypsilantis/Documents/Projects/LLM_Agents/DeepResearch/deep-searcher/data"
load_from_local_files(paths_or_directory=local_path)