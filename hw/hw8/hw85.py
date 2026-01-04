from typing import Tuple, List

def _to_bits4(x) -> List[int]:
    """
    Accepts:
      - 4-bit string like "1011"
      - list/tuple of 4 ints [1,0,1,1]
    Returns list of 4 ints (0/1).
    """
    if isinstance(x, str):
        if len(x) != 4 or any(c not in "01" for c in x):
            raise ValueError("Input must be a 4-bit string like '1011'.")
        return [int(c) for c in x]
    if isinstance(x, (list, tuple)):
        if len(x) != 4 or any(b not in (0, 1) for b in x):
            raise ValueError("Input must be 4 bits like [1,0,1,1].")
        return list(x)
    raise TypeError("Unsupported input type for 4-bit data.")

def _to_bits7(x) -> List[int]:
    """
    Accepts:
      - 7-bit string like "0110011"
      - list/tuple of 7 ints
    Returns list of 7 ints (0/1).
    """
    if isinstance(x, str):
        if len(x) != 7 or any(c not in "01" for c in x):
            raise ValueError("Input must be a 7-bit string like '0110011'.")
        return [int(c) for c in x]
    if isinstance(x, (list, tuple)):
        if len(x) != 7 or any(b not in (0, 1) for b in x):
            raise ValueError("Input must be 7 bits like [0,1,1,0,0,1,1].")
        return list(x)
    raise TypeError("Unsupported input type for 7-bit codeword.")

def hamming74_encode(data4) -> str:
    """
    Encode 4 data bits into Hamming(7,4) codeword (even parity).
    data bits mapped to positions: 3,5,6,7 (d1,d2,d3,d4).
    parity bits at positions: 1,2,4 (p1,p2,p4).
    Returns 7-bit string.
    """
    d1, d2, d3, d4 = _to_bits4(data4)

    # Positions 1..7 (use 1-indexed conceptual mapping)
    b = [None] * 8  # ignore index 0
    b[3], b[5], b[6], b[7] = d1, d2, d3, d4

    # Even parity sets:
    # p1 covers positions with LSB=1: 1,3,5,7  -> p1 = b3 XOR b5 XOR b7
    # p2 covers positions with bit2=1: 2,3,6,7 -> p2 = b3 XOR b6 XOR b7
    # p4 covers positions with bit4=1: 4,5,6,7 -> p4 = b5 XOR b6 XOR b7
    p1 = b[3] ^ b[5] ^ b[7]
    p2 = b[3] ^ b[6] ^ b[7]
    p4 = b[5] ^ b[6] ^ b[7]

    b[1], b[2], b[4] = p1, p2, p4

    return "".join(str(b[i]) for i in range(1, 8))

def hamming74_decode(code7) -> Tuple[str, int, str]:
    """
    Decode (and correct 1-bit error) for Hamming(7,4) with even parity.
    Returns (data4_str, error_position, corrected_code7_str)
      - error_position: 0 means no error detected, else 1..7 indicates corrected bit.
    """
    r = _to_bits7(code7)
    b = [None] + r[:]  # b[1..7]

    # Syndrome bits (even parity check):
    # s1 checks (1,3,5,7), s2 checks (2,3,6,7), s4 checks (4,5,6,7)
    s1 = b[1] ^ b[3] ^ b[5] ^ b[7]
    s2 = b[2] ^ b[3] ^ b[6] ^ b[7]
    s4 = b[4] ^ b[5] ^ b[6] ^ b[7]

    # error position = s1*1 + s2*2 + s4*4
    err_pos = s1 * 1 + s2 * 2 + s4 * 4

    corrected = b[:]
    if err_pos != 0:
        corrected[err_pos] ^= 1  # flip the erroneous bit

    # Extract data bits from positions 3,5,6,7
    data_bits = [corrected[3], corrected[5], corrected[6], corrected[7]]
    data4_str = "".join(str(x) for x in data_bits)
    corrected_code7_str = "".join(str(corrected[i]) for i in range(1, 8))

    return data4_str, err_pos, corrected_code7_str

# ------------------- demo -------------------
if __name__ == "__main__":
    data = "1011"
    code = hamming74_encode(data)
    print("data:", data)
    print("encoded:", code)

    # Introduce 1-bit error (flip position 6 for demo)
    pos = 6
    code_err = list(code)
    code_err[pos - 1] = "1" if code_err[pos - 1] == "0" else "0"
    code_err = "".join(code_err)
    print("with 1-bit error at pos", pos, ":", code_err)

    decoded, err_pos, corrected = hamming74_decode(code_err)
    print("decoded data:", decoded)
    print("detected/corrected error position:", err_pos)
    print("corrected code:", corrected)
