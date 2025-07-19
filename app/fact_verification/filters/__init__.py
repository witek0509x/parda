from .metadata_equals import MetadataEqualsFilter
from .metadata_range import MetadataRangeFilter
from .gene_expression import GeneExpressionFilter
from .keyword_filter import KeywordFilter

ALL_FILTERS = [
    MetadataEqualsFilter,
    MetadataRangeFilter,
    GeneExpressionFilter,
    KeywordFilter,
]

def create_filter(filter_dict):
    """Instantiate appropriate filter class from dict or raise ValueError."""
    if not isinstance(filter_dict, dict):
        raise ValueError("Filter must be a dict")
    kind = filter_dict.get("kind")
    for cls in ALL_FILTERS:
        if cls.kind == kind:
            instance = cls.from_dict(filter_dict)
            if instance is None:
                raise ValueError(f"Invalid parameters for filter kind '{kind}'")
            return instance
    raise ValueError(f"Unknown filter kind '{kind}'")

__all__ = [
    "MetadataEqualsFilter",
    "MetadataRangeFilter",
    "GeneExpressionFilter",
    "KeywordFilter",
    "create_filter",
] 