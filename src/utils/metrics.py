import jiwer


def exact_match(true_text: str, predicted_text: str) -> bool:
    return true_text == predicted_text


def character_error_rate(true_text: str, predicted_text: str) -> float:
    return jiwer.cer(true_text, predicted_text)


def word_error_rate(true_text: str, predicted_text: str) -> float:
    return jiwer.wer(true_text, predicted_text)


def digit_accuracy(true_text: str, predicted_text: str) -> float:
    max_len = max(len(true_text), len(predicted_text))
    if max_len == 0:
        return 1.0

    correct = 0
    for i in range(max_len):
        true_char = true_text[i] if i < len(true_text) else None
        pre_char = predicted_text[i] if i < len(predicted_text) else None
        if true_char == pre_char:
            correct += 1

    return correct / max_len


def numeric_error(true_text: str, predicted_text: str) -> float | None:
    try:
        true_number = float(true_text)
        predicted_number = float(predicted_text)
    except ValueError:
        return None

    return abs(true_number - predicted_number)


def classify_error(true_text: str, predicted_text: str) -> str:
    if true_text == predicted_text:
        return "correct"

    digits_match = true_text.replace(".", "") == predicted_text.replace(".", "")
    dot_match = true_text.find(".") == predicted_text.find(".")

    if digits_match and not dot_match:
        return "dot_only"

    if not digits_match and dot_match:
        return "digits_only"

    return "both"
