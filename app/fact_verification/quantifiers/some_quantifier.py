from __future__ import annotations

from ..base_quantifier import BaseQuantifier, VerificationResult


class SomeQuantifier(BaseQuantifier):
    """Qualifier SOME – at least one but not all filtered cells satisfy condition."""

    name = "SOME"

    @classmethod
    def from_str(cls, qualifier: str) -> "SomeQuantifier":
        if qualifier.upper() != cls.name:
            raise ValueError
        return cls()

    def verify(self, positives: int, total: int) -> VerificationResult:
        if total == 0:
            return VerificationResult.not_verifiable("Filtered all cells")
        ok = 0 < positives
        return VerificationResult.create_true(positives/total) if ok else VerificationResult.create_false(positives/total) 