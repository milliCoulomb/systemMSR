# methods/method.py
import numpy as np
from abc import ABC, abstractmethod

class Method(ABC):
    @abstractmethod
    def build_stif(self, k: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def build_mass(self, m: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def build_grad(self, u: np.ndarray) -> np.ndarray:
        pass