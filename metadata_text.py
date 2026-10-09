"""Store unchanged JSON values without quadratic single-line Text insertion."""
import json


def encode_metadata(metadata, line_limit=2048):
    # Encoder chunks end at JSON token boundaries. Whitespace between them is
    # legal; never split an escaped string or a number to wrap a long line.
    parts = []
    width = 0
    for token in json.JSONEncoder().iterencode(metadata):
        if width and width + len(token) > line_limit:
            parts.append('\n')
            width = 0
        parts.append(token)
        width += len(token)
    return ''.join(parts)


def write_metadata(text, metadata):
    text.from_string(encode_metadata(metadata))
