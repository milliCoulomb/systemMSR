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
    exchanger_length: float
    core_radius: float
    n_cells_core: int
    n_cells_exchanger: int

    def __post_init__(self):
        self.dx_core = self.core_length / self.n_cells_core
        self.dx_exchanger = self.exchanger_length / self.n_cells_exchanger
        self.dx = np.array(
            [self.dx_core] * self.n_cells_core
            + [self.dx_exchanger] * self.n_cells_exchanger
        )
        self.x = np.linspace(
            0,
            self.core_length + self.exchanger_length,
            self.n_cells_core + self.n_cells_exchanger,
        )


@dataclass
class SecondaryLoopGeometry:
    first_loop_length: float
    exchanger_length: float
    second_loop_length: float
    loop_radius: float
    n_cells_first_loop: int
    n_cells_exchanger: int
    n_cells_second_loop: int

    def __post_init__(self):
        self.dx_first_loop = self.first_loop_length / self.n_cells_first_loop
        self.dx_exchanger = self.exchanger_length / self.n_cells_exchanger
        self.dx_second_loop = self.second_loop_length / self.n_cells_second_loop
        self.dx = np.array(
            [self.dx_first_loop] * self.n_cells_first_loop
            + [self.dx_exchanger] * self.n_cells_exchanger
            + [self.dx_second_loop] * self.n_cells_second_loop
        )
