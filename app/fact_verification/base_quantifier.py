from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum, auto


class VerificationResult():
    veryfiable = False
    is_true = False
    true_ratio = 0
    reason = ""

    def __init__(self, veryfiable, is_true, true_ratio, reason):
        self.veryfiable = veryfiable
        self.is_true = is_true
        self.true_ratio = true_ratio
        self.reason = reason        

    @staticmethod
    def create_true(count: int):
        return VerificationResult(veryfiable=True, is_true=True, true_ratio=count, reason="")
    
    @staticmethod
    def create_false(count: int):
        return VerificationResult(veryfiable=True, is_true=False, true_ratio=count, reason="")
    
    @staticmethod
    def not_verifiable(reason: str):
        return VerificationResult(veryfiable=False, is_true=False, true_ratio=0, reason=reason)


class BaseQuantifier(ABC):
    """Decides whether the number of positive cells satisfies the claim."""

    name: str = "abstract"

    @classmethod
    @abstractmethod
    def from_str(cls, qualifier: str) -> "BaseQuantifier":
        """Return concrete quantifier matching *qualifier* string or raise ValueError."""

    @abstractmethod
    def verify(self, positives: int, total: int) -> VerificationResult:
        """Return verification outcome for a given count of positives out of *total* filtered cells.""" 