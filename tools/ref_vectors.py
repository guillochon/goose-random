"""Reference outputs for test/test_random.goose, from independent Python
implementations of the generators (and, for SFC64, NumPy's own).

    python tools/ref_vectors.py
"""
import numpy as np

M = (1 << 64) - 1


def rotl(x, k):
    return ((x << k) | (x >> (64 - k))) & M


def splitmix(s):
    while True:
        s = (s + 0x9E3779B97F4A7C15) & M
        z = s
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & M
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & M
        yield z ^ (z >> 31)


class Xoshiro:
    def __init__(self, s, star):
        self.s = list(s)
        self.star = star

    def next(self):
        s = self.s
        if self.star:
            r = (rotl((s[1] * 5) & M, 7) * 9) & M
        else:
            r = (rotl((s[0] + s[3]) & M, 23) + s[0]) & M
        t = (s[1] << 17) & M
        s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3]; s[2] ^= t
        s[3] = rotl(s[3], 45)
        return r

    def jump(self, poly):
        acc = [0, 0, 0, 0]
        for w in poly:
            for b in range(64):
                if (w >> b) & 1:
                    acc = [a ^ x for a, x in zip(acc, self.s)]
                self.next()
        self.s = acc


JUMP = [0x180EC6D33CFD0ABA, 0xD5A61266F0C9392C, 0xA9582618E03FC9AA, 0x39ABDC4529B1661C]
LONG_JUMP = [0x76E15D3EFEFDCBBF, 0xC5004E441C522FB3, 0x77710069854EE241, 0x39109BB02ACBE635]


def sfc64_raw(a, b, c, n):
    w = 1
    out = []
    for i in range(12 + n):
        r = (a + b + w) & M
        w += 1
        a = b ^ (b >> 11)
        b = (c + (c << 3)) & M
        c = (rotl(c, 24) + r) & M
        if i >= 12:
            out.append(r)
    return out


def show(name, vals):
    print(f'{name}: [' + ', '.join(str(v) for v in vals) + ']')


def main():
    sm = splitmix(1234567)
    show('splitmix64(1234567)', [next(sm) for _ in range(5)])

    for star in (False, True):
        tag = 'ss' if star else 'pp'
        g = Xoshiro([1, 2, 3, 4], star)
        show(f'xoshiro256{tag} {{1,2,3,4}}', [g.next() for _ in range(6)])
        sm = splitmix(42)
        g = Xoshiro([next(sm) for _ in range(4)], star)
        show(f'xoshiro256{tag} seed(42)', [g.next() for _ in range(4)])
        g = Xoshiro([1, 2, 3, 4], star)
        g.jump(JUMP)
        show(f'xoshiro256{tag} {{1,2,3,4}} jump', [g.next() for _ in range(3)])
        g = Xoshiro([1, 2, 3, 4], star)
        g.jump(LONG_JUMP)
        show(f'xoshiro256{tag} {{1,2,3,4}} long_jump', [g.next() for _ in range(3)])

    words = (0x0123456789ABCDEF, 0xFEDCBA9876543210, 0x0F1E2D3C4B5A6978)
    mine = sfc64_raw(*words, 6)
    bg = np.random.SFC64()
    st = bg.state
    st['state']['state'] = np.array(list(words) + [1], dtype=np.uint64)
    st['has_uint32'] = 0
    bg.state = st
    for _ in range(12):
        bg.random_raw()
    theirs = [int(x) for x in bg.random_raw(6)]
    assert mine == theirs, (mine, theirs)
    show('sfc64 words (NumPy agrees)', mine)


if __name__ == '__main__':
    main()
