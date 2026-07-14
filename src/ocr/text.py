VALID_CHARS = set("0123456789.")


def encode_label(text: str) -> str:
    return " ".join(str(text).strip())


def decode_label(text: str) -> str:
    return "".join(str(text).split())


def is_valid_reading(text: str) -> bool:
    text = str(text)

    if not text or set(text) - VALID_CHARS:
        return False

    if text.count(".") > 1:
        return False

    return not text.startswith(".") and not text.endswith(".")
