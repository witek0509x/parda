from __future__ import annotations

from typing import Dict, Type

from .all_quantifier import AllQuantifier
from .none_quantifier import NoneQuantifier
from .some_quantifier import SomeQuantifier

from ..base_quantifier import BaseQuantifier

_QUALIFIER_MAP = {
    AllQuantifier.name: AllQuantifier,
    NoneQuantifier.name: NoneQuantifier,
    SomeQuantifier.name: SomeQuantifier,
}

def create_quantifier(q: str) -> BaseQuantifier:
    key = q.upper()
    if key not in _QUALIFIER_MAP:
        raise ValueError(f"Unknown qualifier '{q}'")
    return _QUALIFIER_MAP[key]()

__all__ = [
    "AllQuantifier",
    "NoneQuantifier",
    "SomeQuantifier",
    "create_quantifier",
] 