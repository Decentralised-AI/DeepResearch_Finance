import sys
import os

# Get the parent directory path
#parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))

# Add parent directory to sys.path if not already present
#if parent_dir not in sys.path:
#    sys.path.insert(0, parent_dir)



from deepsearcher.configuration import Configuration, init_config
from deepsearcher.online_query import query