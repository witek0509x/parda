from __future__ import annotations

from ..base_quantifier import BaseQuantifier, VerificationResult


class NoneQuantifier(BaseQuantifier):
    """Qualifier NONE – none of the filtered cells satisfy condition."""

    name = "NONE"

    @classmethod
    def from_str(cls, qualifier: str) -> "NoneQuantifier":
        if qualifier.upper() != cls.name:
            raise ValueError
        return cls()

    def verify(self, positives: int, total: int) -> VerificationResult:
        if total == 0:
            return VerificationResult.not_verifiable("Filtered all cells")
        return VerificationResult.create_true(0) if positives == 0 else VerificationResult.create_false(0) 