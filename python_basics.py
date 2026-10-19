"""Заготовки задач на базовый Python."""

from grader_contracts.python_basics import PositiveIntegerInput, TextInput, VectorPairInput


def count_vowels(data: TextInput) -> int:
    text = data.value
    vowels = set("aeiouAEIOU")
    return sum(1 for char in text if char in vowels)


def has_unique_characters(data: TextInput) -> bool:
    text = data.value
    return len(text) == len(set(text))


def count_one_bits(data: PositiveIntegerInput) -> int:
    number = data.value
    return number.bit_count()


import math


def multiplicative_persistence(data: PositiveIntegerInput) -> int:
    number = data.value
    steps = 0

    while number >= 10:
        number = math.prod(int(digit) for digit in str(number))
        steps += 1

    return steps


def mse(data: VectorPairInput) -> float:
    predicted, expected = data.predicted, data.expected
    if not predicted:
        return 0.0

    total_squared_error = sum((p - e) ** 2 for p, e in zip(predicted, expected))
    return float(total_squared_error / len(predicted))


def prime_factorization(data: PositiveIntegerInput) -> str:
    number = data.value
    if number == 1:
        return "(1)"

    factors = []
    d = 2

    while d * d <= number:
        if number % d == 0:
            count = 0
            while number % d == 0:
                count += 1
                number //= d
            if count == 1:
                factors.append(f"({d})")
            else:
                factors.append(f"({d}**{count})")
        d += 1

    if number > 1:
        factors.append(f"({number})")

    return "".join(factors)


def pyramid(data: PositiveIntegerInput) -> int | str:
    cube_count = data.value

    current_sum = 0
    k = 1

    while current_sum < cube_count:
        current_sum += k * k
        if current_sum == cube_count:
            return k
        k += 1

    return "It is impossible"


def is_balanced_number(data: PositiveIntegerInput) -> bool:
    number = data.value
    digits = str(number)
    n = len(digits)

    if n <= 2:
        return True

    side_len = (n - 1) // 2

    left_sum = sum(int(d) for d in digits[:side_len])
    right_sum = sum(int(d) for d in digits[-side_len:])

    return left_sum == right_sum
