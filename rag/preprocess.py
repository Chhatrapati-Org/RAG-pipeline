import re

# Compile regex patterns once for better performance
_REMOVAL_PATTERN = re.compile(
    r"\"\w+\"\s*:\s*(?:null|None)"  # JSON null values
    r"|\"\S{,15}\"\s*:"              # Short JSON keys
    r"|/url\?q=\S+"                  # URL query params
    r"|https?://\S+"                 # HTTP/HTTPS URLs
    r"|www\.\S+"                     # www URLs
    r"|<[^>]+>"                      # HTML tags
    r"|\\n|\\\\|\\\"|\\'"            # Escape sequences
    r"|\"",                          # Quotes
    re.IGNORECASE
)

_WHITESPACE_PATTERN = re.compile(r"\s+")
_COMMA_CLEANUP = re.compile(r"\s*,\s*,+")
_PERIOD_CLEANUP = re.compile(r"\s*\.\s*\.+")
_MIXED_PUNCT = re.compile(r"(?:,\.)|(?:\.,)")


def preprocess_chunk_text(text: str) -> str:
    """
    Preprocess text by removing unwanted patterns and normalizing whitespace.
    Optimized with pre-compiled regex patterns for better performance.
    """
    # Single pass removal of all unwanted patterns
    text = _REMOVAL_PATTERN.sub(" ", text)
    
    # Normalize whitespace
    text = _WHITESPACE_PATTERN.sub(" ", text)
    
    # Clean up punctuation
    text = _COMMA_CLEANUP.sub(", ", text)
    text = _PERIOD_CLEANUP.sub(". ", text)
    text = _MIXED_PUNCT.sub(". ", text)
    
    # Final whitespace normalization and strip
    text = _WHITESPACE_PATTERN.sub(" ", text)
    text = text.strip()

    return text
