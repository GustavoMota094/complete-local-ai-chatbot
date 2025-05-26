import hashlib
import logging

logger = logging.getLogger(__name__)

def create_safe_redis_identifier(input_string: str) -> str:
    """
    Creates a safe identifier suitable for Redis keys or other systems
    that might have restrictions on identifier characters or length.

    Uses SHA-1 hashing to generate a fixed-length hexadecimal string.

    Args:
        input_string: The original string identifier (e.g., session ID).

    Returns:
        A safe, fixed-length hexadecimal string identifier derived from the input.

    Raises:
        ValueError: If the input string cannot be processed or hashing fails.
    """
    if not isinstance(input_string, str):
        logger.warning(f"Input for safe identifier creation is not a string (type: {type(input_string)}). Attempting conversion.")
        input_string = str(input_string)

    if not input_string:
        logger.warning("Attempted to create safe identifier from an empty string. Using a placeholder.")
        input_string = "empty_input_placeholder"

    try:
        input_bytes = input_string.encode('utf-8')
        hasher = hashlib.sha1(input_bytes)
        safe_id = hasher.hexdigest()
        logger.debug(f"Generated safe ID '{safe_id}' for input '{input_string}'")
        return safe_id
    except UnicodeEncodeError as e:
        logger.error(f"Failed to encode input string to UTF-8 for hashing: '{input_string[:50]}...': {e}", exc_info=True)
        raise ValueError(f"Failed to generate safe identifier due to encoding error for input: {input_string[:50]}...") from e
    except Exception as e:
        logger.error(f"Failed to hash input string '{input_string[:50]}...': {e}", exc_info=True)
        raise ValueError(f"Failed to generate safe identifier from input: {input_string[:50]}...") from e