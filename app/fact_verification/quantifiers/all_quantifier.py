from __future__ import annotations

from ..base_quantifier import BaseQuantifier, VerificationResult


class AllQuantifier(BaseQuantifier):
    """Qualifier ALL – all filtered cells must satisfy condition."""

    name = "ALL"

    @classmethod
    def from_str(cls, qualifier: str) -> "AllQuantifier":
        if qualifier.upper() != cls.name:
            raise ValueError
        return cls()

    # No internal state needed

    def verify(self, positives: int, total: int) -> VerificationResult:  # noqa: D401
        if total == 0:
            return VerificationResult.not_verifiable("Filtered all cells")
        return VerificationResult.create_true(positives/total) if positives == total else VerificationResult.create_false(positives/total) 