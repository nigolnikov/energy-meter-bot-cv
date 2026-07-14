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
