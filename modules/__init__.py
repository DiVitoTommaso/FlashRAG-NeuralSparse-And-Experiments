from .modules_logger import logger

from flashrag.config import *
from flashrag.prompt import *
from flashrag.retriever import *
from flashrag.utils import *

import os
import gc
import io
import sys
import random
import time

import socket
import struct
import json
import ast

from .slide_window_module import *
from .rag_evaluation_module import *
from .prompt_refiner_module import *
from .generator_module import *
from .noise_module import *
from .reorder_module import *
from .prompt_module import *
from .rerank_module import *
from .dataset_module import *
from .retriever_module import *
from .sock_remote_module import *
from .mpi_remote_module import *

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def gpu_release():
    """
    Clear Python and GPU memory caches to help avoid memory leaks and
    release unused memory back to the system.
    """
    gc.collect()  # Run Python garbage collector

    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # Clear GPU cached memory
        torch.cuda.reset_peak_memory_stats()  # Reset peak memory tracking stats

    gc.collect()  # Run garbage collector again to finalize cleanup

    time.sleep(1)  # Sleep 1 second to give GPU time to release memory properly