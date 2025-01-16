# utils/geometry.py
# a class that contains the geometry of the reactor core and heat exchanger

from dataclasses import dataclass, field
from typing import List, Dict
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class CoreGeometry:
    core_length: float
    exchange_length: float
    core_radius: float
    n_cells_core: int
    n_cells_exchanger: int

@dataclass
class SecondaryLoopGeometry:
    first_loop_length: float
    exchange_length: float
    second_loop_length: float
    loop_radius: float
    n_cells_first_loop: int
    n_cells_exchange: int
    n_cells_second_loop: int