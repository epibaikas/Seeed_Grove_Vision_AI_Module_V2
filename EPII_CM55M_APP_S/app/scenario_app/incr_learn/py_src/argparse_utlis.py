import argparse

def positive_int(s: str) -> int:
    try:
        v = int(s)
    except ValueError:
        raise argparse.ArgumentTypeError(f'Expected integer, got {s!r}')

    if v <= 0:
        raise argparse.ArgumentTypeError(f'Expected positive integer, got {v}')

    return v