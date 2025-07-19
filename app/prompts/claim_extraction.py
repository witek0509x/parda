FILTER_DEFINITION = (
    "Filters optionally narrow which cells the claim applies to. Provide zero or more filters.\n"\
    "Supported filter objects:\n"\
    "  • metadata_equals  → { 'kind': 'metadata_equals',  'key': <string>, 'value': <string> }\n"\
    "  • metadata_range   → { 'kind': 'metadata_range',   'key': <string>, 'low': <number>, 'high': <number> }\n"\
    "  • gene_expression  → { 'kind': 'gene_expression', 'gene': <string>, 'op': '>', 'value': <number> }\n"\
    "  • keyword          → { 'kind': 'keyword',         'value': 'NONE' | 'SELECTED' | 'HIGHLIGHTED' (this is very important filter, ensure that if agent refers to selected or highlighted cells, you will include this in filter list, also if it is only contextually infered)]}\n"
)

QUALIFIER_INFO = (
    "Each claim must include a 'qualifier' field that tells how many of the filtered cells satisfy the assertion:\n"\
    "  • 'ALL'   – every filtered cell meets the condition.\n"\
    "  • 'NONE'  – no filtered cell meets the condition.\n"\
    "  • 'SOME'  – at least one but not all filtered cells meet the condition.\n"
)

COMMON_FIELDS = (
    "Common required fields for every claim JSON object:\n"\
    "  'type'      – string, one of the supported claim types.\n"\
    "  'filters'   – array of zero or more filter objects (see above).\n"\
    "  'qualifier' – 'ALL' | 'NONE' | 'SOME'.\n"\
    "  'text'      – short natural-language restatement of the claim for UI display.\n"
)

GENE_EXPRESSION_TEMPLATE = (
    "--- Gene Expression Claim Template ---\n"\
    "Specific fields for this type:\n"\
    "  'gene'  – gene symbol.\n"\
    "  'op'    – one of '>' '<' '>=' '<=' '=='.\n"\
    "  'value' – numeric threshold.\n"\
    "Example:\n"\
    "{\n"\
    "  'type': 'gene_expression',\n"\
    "  'filters': [],\n"\
    "  'qualifier': 'ALL',\n"\
    "  'gene': 'GeneName0',\n"\
    "  'op': '>',\n"\
    "  'value': 0,\n"\
    "  'text': 'All cells express GeneName0.'\n"\
    "}\n"
)

METADATA_EQUALS_TEMPLATE = (
    "--- Metadata Equals Claim Template ---\n"\
    "When referring to metadata columns you MUST use the exact column names and metadata values as they are given in the dataset (refer to dataset summary for value names, try to find most similar one to the one from assistant claim) (including any underscores, dashes, or other characters).\n"\
    "Specific fields for this type:\n"\
    "  'key'   – metadata column.\n"\
    "  'value' – expected exact value.\n"\
    "Template:\n"\
    "{\n"\
    "  'type': 'metadata_equals',\n"\
    "  'filters': [ { 'kind': 'keyword', 'value': 'SELECTED' } ],\n"\
    "  'qualifier': 'NONE',\n"\
    "  'key': 'treatment',\n"\
    "  'value': 'drugA',\n"\
    "  'text': 'None of the selected cells are drugA treated.'\n"\
    "}\n"
)

METADATA_RANGE_TEMPLATE = (
    "--- Metadata Range Claim Template ---\n"\
    "Specific fields for this type:\n"\
    "  'key'  – metadata column.\n"\
    "  'low'  – inclusive lower bound.\n"\
    "  'high' – inclusive upper bound.\n"\
    "Template:\n"\
    "{\n"\
    "  'type': 'metadata_range',\n"\
    "  'filters': [],\n"\
    "  'qualifier': 'SOME',\n"\
    "  'key': 'n_counts',\n"\
    "  'low': 5000,\n"\
    "  'high': 10000,\n"\
    "  'text': 'Some cells have n_counts between 5k and 10k.'\n"\
    "}\n"
)

FULL_PROMPT = (
    "You are an information-extraction assistant. Read the assistant response enclosed in triple backticks.\n"\
    "Identify every distinct claim about the scRNA-seq dataset, and output them as a PURE JSON ARRAY (no markdown).\n"\
    "Each claim object MUST follow this rule set EXACTLY – missing or additional keys are errors.\n\n"\
    + FILTER_DEFINITION + "\n" + QUALIFIER_INFO + "\n" + COMMON_FIELDS + "\n" +
    GENE_EXPRESSION_TEMPLATE + "\n" + METADATA_EQUALS_TEMPLATE + "\n" + METADATA_RANGE_TEMPLATE + "\n" +
    "If no data-related claims are present, output []."
) 