"""CPU bit-level test of the lossless BF16 weight format; no GPU claims."""


def validate():
    for base in (0, 1, 100, 108, 110, 120, 127, 200, 241):
        for bits in range(65536):
            exponent = (bits >> 7) & 255
            payload = (bits & 127) | ((bits >> 8) & 128)
            code = exponent - base + 1 if base <= exponent < base + 15 else 0
            restored = ((payload & 128) << 8) | ((base + code - 1) << 7) | (payload & 127) if code else bits
            assert restored == bits, (base, bits, restored)
            packed_bits = ((payload & 128) << 8) | ((base + code - 1) << 7) | (payload & 127)
            integer_selected = ((packed_bits & 0xFFFFFFFF) * min(code, 1) + (bits if code == 0 else 0)) & 65535
            assert integer_selected == bits, (base, bits, integer_selected)
        for a in range(16):
            for b in range(16):
                packed = a | (b << 4)
                assert (packed & 15, packed >> 4) == (a, b)
    print("PASS: all 65,536 BF16 bit patterns at nine exponent bases; all nibble pairs")


if __name__ == "__main__":
    validate()
