# Код взят отсюда: https://gist.github.com/Lipen/0dfabf4a3b069e0539cdb809136eca68

from typing import List


def str2bits(s: str, n=0) -> List[bool]:
    n = max(len(s), n) - len(s)
    s = str('0' * n) + s
    return [{'1': True, '0': False}[c] for c in s]


def bits2str(bits: List[bool]) -> str:
    return ''.join(str(int(b)) for b in bits)


def num2str(x: int, n: int) -> str:
    return f"{x:0{n}b}"


def num2bits(x: int, n: int):
    return str2bits(num2str(x, n))


def bits2num(bits: List[bool]) -> int:
    return int(bits2str(bits), 2)


def encode_geq(
        x: List[int],
        a: List[bool],
) -> List[List[int]]:
    assert len(x) == len(a)

    if len(x) == 0:
        return []

    clauses = []
    assert isinstance(a[0], bool)
    if a[0]:
        clauses.append([x[0]])
        clauses.extend(encode_geq(x[1:], a[1:]))
    else:
        # Append (x=1) to all sub-clauses:
        for clause in encode_geq(x[1:], a[1:]):
            clauses.append([x[0]] + clause)
    return clauses


def encode_leq(
        x: List[int],
        b: List[bool],
) -> List[List[int]]:
    assert len(x) == len(b)

    if len(x) == 0:
        return []

    clauses = []
    assert isinstance(b[0], bool)
    if not b[0]:
        clauses.append([-x[0]])
        clauses.extend(encode_leq(x[1:], b[1:]))
    else:
        # Append (x=0) to all sub-clauses:
        for clause in encode_leq(x[1:], b[1:]):
            clauses.append([-x[0]] + clause)
    return clauses


def encode_geq_and_leq(
        x: List[int],
        a: List[bool],
        b: List[bool],
) -> List[List[int]]:
    assert len(x) == len(a)
    assert len(x) == len(b)

    if len(x) == 0:
        return []

    clauses = []
    assert isinstance(a[0], bool)
    assert isinstance(b[0], bool)
    if a[0]:
        assert b[0]
        clauses.append([x[0]])
        clauses.extend(encode_geq_and_leq(x[1:], a[1:], b[1:]))
    elif not b[0]:
        assert not a[0]
        clauses.append([-x[0]])
        clauses.extend(encode_geq_and_leq(x[1:], a[1:], b[1:]))
    else:
        assert not a[0]
        assert b[0]
        # Append (x=1) to all sub-clauses:
        for clause in encode_geq(x[1:], a[1:]):
            clauses.append([x[0]] + clause)
            # Append (x=0) to all sub-clauses:
        for clause in encode_leq(x[1:], b[1:]):
            clauses.append([-x[0]] + clause)
    return clauses
