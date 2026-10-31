// random: random number generators and distributions in pure Goose.
//
// Generators are small flat structs passed by reference: xoshiro256++ (the
// default, `random::seed(42)`), xoshiro256**, SFC64 and splitmix64, plus
// std's `rng`. Every distribution is generic over the generator, so
// `random::normal(g)` works with any of them, and a generator of your own
// joins in by declaring `fn random::next_u64(g: mine&) -> u64`.
//
// Accuracy first, then speed:
//
// * Uniform doubles carry 53 random bits; `uniform_dense` reaches every
//   double in (0, 1) with its exact probability.
// * Integers in a range are exactly uniform (Lemire's multiply-and-reject).
// * The normal and exponential samplers are 256-layer ziggurats whose
//   tables are the correctly rounded values of their definitions
//   (tools/gen_tables.py), with exact tails and no reuse of the bits that
//   choose the layer. Most draws cost one 64-bit output, a table lookup and
//   a compare.
// * The other distributions are exact algorithms (rejection or inversion
//   with exact arithmetic where it matters), chosen per parameter range:
//   Marsaglia-Tsang gamma, Hormann's PTRS Poisson and BTRS binomial,
//   inversion from the mode for the hypergeometric, and the rest built from
//   these. Tails come from the ziggurats or from logs of uniforms that never
//   reach zero, so they are not cut off at a few dozen standard deviations.
//
// Parallel streams: `split(g)` returns a generator and jumps g ahead by
// 2^128 outputs, so the streams it hands out never overlap.

import std;
import math;
import os;
import dictionary;
import tables;

namespace random;

// ---------------------------------------------------------------------------
// Generators

// xoshiro256++ (Blackman and Vigna, 2019): 256 bits of state, period
// 2^256 - 1, no multiplications, and passes every test in BigCrush and
// PractRand. The default.
struct xoshiro256pp { s0: u64, s1: u64, s2: u64, s3: u64 }

// xoshiro256**: the same engine with a multiplying scrambler.
struct xoshiro256ss { s0: u64, s1: u64, s2: u64, s3: u64 }

// SFC64 (Chris Doty-Humphrey): a chaotic generator with a counter, so its
// period is at least 2^64; the fastest here on many machines. Its output is
// NumPy's `SFC64` for the same state.
struct sfc64 { a: u64, b: u64, c: u64, w: u64 }

// splitmix64 (Steele, Lea and Flood, 2014): one word of state. Fine on its
// own, and the standard way to expand a 64-bit seed into a bigger state.
struct splitmix64 { s: u64 }

inline fn rotl(x: u64, k: u64) -> u64 { (x << k) | (x >> (64 - k)) }

inline fn next_u64(g: xoshiro256pp&) -> u64 {
    let r = rotl(g.s0 + g.s3, 23) + g.s0;
    let t = g.s1 << 17;
    g.s2 ^= g.s0;
    g.s3 ^= g.s1;
    g.s1 ^= g.s2;
    g.s0 ^= g.s3;
    g.s2 ^= t;
    g.s3 = rotl(g.s3, 45);
    return r;
}

inline fn next_u64(g: xoshiro256ss&) -> u64 {
    let r = rotl(g.s1 * 5, 7) * 9;
    let t = g.s1 << 17;
    g.s2 ^= g.s0;
    g.s3 ^= g.s1;
    g.s1 ^= g.s2;
    g.s0 ^= g.s3;
    g.s2 ^= t;
    g.s3 = rotl(g.s3, 45);
    return r;
}

inline fn next_u64(g: sfc64&) -> u64 {
    let r = g.a + g.b + g.w;
    g.w += 1;
    g.a = g.b ^ (g.b >> 11);
    g.b = g.c + (g.c << 3);
    g.c = rotl(g.c, 24) + r;
    return r;
}

inline fn next_u64(g: splitmix64&) -> u64 {
    g.s += 0x9E3779B97F4A7C15;
    var z = g.s;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB;
    return z ^ (z >> 31);
}

// std's splitmix64 `rng`, so the distributions take it too.
inline fn next_u64(g: ::rng&) -> u64 { ::rand_u64(g) }

// The high 32 bits, the better ones of every generator here.
inline fn next_u32(g: xoshiro256pp&) -> u32 { (next_u64(g) >> 32) as u32 }

// ---------------------------------------------------------------------------
// Seeding and streams

// The default generator, its state filled from s by splitmix64, as the
// xoshiro authors recommend. Different seeds give unrelated streams.
fn seed(s: u64) -> xoshiro256pp {
    var m = splitmix64 { s };
    return xoshiro256pp { next_u64(m), next_u64(m), next_u64(m), next_u64(m) };
}

// The default generator seeded from the operating system's entropy.
fn seed_from_entropy() -> xoshiro256pp { seed(::random_seed()) }

// Seeding for each generator, from one 64-bit seed through splitmix64.
fn seed_xoshiro256ss(s: u64) -> xoshiro256ss {
    var m = splitmix64 { s };
    return xoshiro256ss { next_u64(m), next_u64(m), next_u64(m), next_u64(m) };
}

// As NumPy seeds SFC64 from three words: a, b and c are the words, the
// counter starts at 1, and 12 outputs are discarded.
fn seed_sfc64(s: u64) -> sfc64 {
    var m = splitmix64 { s };
    return sfc64_from_words(next_u64(m), next_u64(m), next_u64(m));
}

fn sfc64_from_words(a: u64, b: u64, c: u64) -> sfc64 {
    var g = sfc64 { a, b, c, 1 };
    for i in 0..12 { next_u64(g); }
    return g;
}

fn seed_splitmix64(s: u64) -> splitmix64 { splitmix64 { s } }

// Advances a xoshiro generator by 2^128 outputs (jump) or 2^192 (long_jump):
// the polynomials of the reference implementation.
let JUMP:      u64[4] = [0x180EC6D33CFD0ABA, 0xD5A61266F0C9392C, 0xA9582618E03FC9AA, 0x39ABDC4529B1661C];
let LONG_JUMP: u64[4] = [0x76E15D3EFEFDCBBF, 0xC5004E441C522FB3, 0x77710069854EE241, 0x39109BB02ACBE635];

fn jump_by<S>(g: S&, poly: u64[4]) {
    var s0: u64 = 0;
    var s1: u64 = 0;
    var s2: u64 = 0;
    var s3: u64 = 0;
    for w in poly {
        for b in 0..64 {
            if (w >> (b as u64)) & 1 != 0 {
                s0 ^= g.s0;
                s1 ^= g.s1;
                s2 ^= g.s2;
                s3 ^= g.s3;
            }
            next_u64(g);
        }
    }
    g.s0 = s0;
    g.s1 = s1;
    g.s2 = s2;
    g.s3 = s3;
}

fn jump(g: xoshiro256pp&) { jump_by(g, JUMP); }
fn jump(g: xoshiro256ss&) { jump_by(g, JUMP); }
fn long_jump(g: xoshiro256pp&) { jump_by(g, LONG_JUMP); }
fn long_jump(g: xoshiro256ss&) { jump_by(g, LONG_JUMP); }

// A copy of g, after which g jumps 2^128 outputs ahead: each call hands out
// a stream that no other one reaches for 2^128 outputs. For one generator
// per thread, call it once per thread from the main one.
fn split(g: xoshiro256pp&) -> xoshiro256pp {
    let r = xoshiro256pp { g.s0, g.s1, g.s2, g.s3 };
    jump(g);
    return r;
}
fn split(g: xoshiro256ss&) -> xoshiro256ss {
    let r = xoshiro256ss { g.s0, g.s1, g.s2, g.s3 };
    jump(g);
    return r;
}

// ---------------------------------------------------------------------------
// Uniform

let TWO_M53 = 1.0 / 9007199254740992.0;    // 2^-53

// Integers below 2^63 become floats through i64 (`as! i64`): x86-64 has an
// instruction for that conversion and none for u64's before AVX-512.
let TWO_M64 = TWO_M53 / 2048.0;             // 2^-64

// In [0, 1): the top 53 bits as a multiple of 2^-53, every one equally
// likely. NumPy's `random()`.
inline fn uniform<G>(g: G&) -> f64 { ((next_u64(g) >> 11) as! i64) * TWO_M53 }

// In [lo, hi).
inline fn uniform<G>(g: G&, lo: f64, hi: f64) -> f64 { lo + (hi - lo) * uniform(g) }

// In (0, 1]: never zero, so its log is finite.
inline fn uniform_pos<G>(g: G&) -> f64 { (((next_u64(g) >> 11) + 1) as! i64) * TWO_M53 }

// In (0, 1), symmetric: the midpoints (k + 1/2) 2^-53.
inline fn uniform_open<G>(g: G&) -> f64 { (((next_u64(g) >> 11) as! i64) + 0.5) * TWO_M53 }

// In (0, 1), reaching every double there, each with the probability of the
// real interval it rounds from (Downey, 2007): the exponent from the count
// of leading zero bits, so values near zero keep all 53 bits. Two outputs
// per value; for inversions whose tails need values below 2^-53.
fn uniform_dense<G>(g: G&) -> f64 {
    var scale = 0.5;            // 2^exponent
    var r = next_u64(g);
    while r == 0 {
        scale *= TWO_M64;
        if scale == 0.0 { return 5e-324; }
        r = next_u64(g);
    }
    let z = clz(r);
    var i = 0;
    while i < z { scale *= 0.5; i++; }
    let m = next_u64(g) >> 12;      // 52 mantissa bits
    // With a zero mantissa the value is the bottom of its binade, whose
    // rounding interval is half in the binade below: half the time it
    // belongs one exponent up.
    if m == 0 && (next_u64(g) >> 63) != 0 { scale *= 2.0; }
    return (1.0 + (m as! i64) * (1.0 / 4503599627370496.0)) * scale;
}

// In [0, 1) as an f32: the top 24 bits as a multiple of 2^-24.
inline fn uniform_f32<G>(g: G&) -> f32 { ((next_u64(g) >> 40) as! i32 as f32) * (1.0 / 16777216.0) }

// ---------------------------------------------------------------------------
// Integers and booleans

// The high and low words of the 128-bit product a * b.
inline fn mul_wide(a: u64, b: u64) -> u64, u64 {
    let a0 = a & 0xFFFFFFFF;
    let a1 = a >> 32;
    let b0 = b & 0xFFFFFFFF;
    let b1 = b >> 32;
    let p00 = a0 * b0;
    let p01 = a0 * b1;
    let p10 = a1 * b0;
    let p11 = a1 * b1;
    let mid = (p00 >> 32) + (p01 & 0xFFFFFFFF) + (p10 & 0xFFFFFFFF);
    return p11 + (p01 >> 32) + (p10 >> 32) + (mid >> 32), a * b;
}

// Exactly uniform in [0, n) for n > 0 (Lemire, 2019): the high word of a
// random multiple of n, drawn again in the rare case the low word lands in
// the 2^64 mod n values that would favor some results. Below 2^32 this
// needs only a 64-bit multiply of the top 32 bits. The common path is
// inline; the rejection step, reached with probability below n / 2^32 (or
// n / 2^64), is not.
inline fn below_u64<G>(g: G&, n: u64) -> u64 {
    assert(n > 0);
    if n <= 0xFFFFFFFF {
        let m = (next_u64(g) >> 32) * n;
        if (m & 0xFFFFFFFF) >= n { return m >> 32; }
        return below_u32_reject(g, n, m);
    }
    let hi, lo = mul_wide(next_u64(g), n);
    if lo >= n { return hi; }
    return below_u64_reject(g, n, hi, lo);
}

fn below_u32_reject<G>(g: G&, n: u64, m0: u64) -> u64 {
    var m = m0;
    let t = (0x100000000 - n) % n;              // 2^32 mod n
    while (m & 0xFFFFFFFF) < t { m = (next_u64(g) >> 32) * n; }
    return m >> 32;
}

fn below_u64_reject<G>(g: G&, n: u64, hi0: u64, lo0: u64) -> u64 {
    var hi = hi0;
    var lo = lo0;
    let t = (~n + 1) % n;                       // 2^64 mod n
    while lo < t { hi, lo = mul_wide(next_u64(g), n); }
    return hi;
}

// Uniform in [0, n), n > 0.
inline fn below<G>(g: G&, n: i64) -> i64 {
    assert(n > 0);
    return below_u64(g, n as u64) as i64;
}

// Uniform in [lo, hi], both included.
fn between<G>(g: G&, lo: i64, hi: i64) -> i64 {
    assert(lo <= hi);
    let span = (hi as! u64) - (lo as! u64);         // hi - lo, mod 2^64
    if span == 0xFFFFFFFFFFFFFFFF { return next_u64(g) as! i64; }
    return ((lo as! u64) + below_u64(g, span + 1)) as! i64;
}

// True with probability p, for p a multiple of 2^-53 exactly.
inline fn bernoulli<G>(g: G&, p: f64) -> bool { uniform(g) < p }

// A fair coin: the top bit.
inline fn coin<G>(g: G&) -> bool { (next_u64(g) >> 63) != 0 }

// +1 or -1, each with probability 1/2, by a lookup rather than a branch that
// mispredicts half the time.
let SIGN: f64[2] = [1.0, -1.0];
inline fn sign<G>(g: G&) -> f64 { SIGN[(next_u64(g) >> 63) as! i64] }

// ---------------------------------------------------------------------------
// Special functions the samplers need, accurate to a few units in the last
// place.

// log(1 + x), accurate for small x (Goldberg's correction: the rounding of
// 1 + x is divided back out).
fn log1p(x: f64) -> f64 {
    let u = 1.0 + x;
    if u == 1.0 { return x; }
    return log(u) * x / (u - 1.0);
}

// exp(x) - 1, accurate for small x (Kahan).
fn expm1(x: f64) -> f64 {
    let u = exp(x);
    if u == 1.0 { return x; }
    let um1 = u - 1.0;
    if um1 == -1.0 { return -1.0; }
    if !is_finite(u) { return u; }
    return um1 * x / log(u);
}

// log(k!) for k >= 0: the table below 128, and Stirling's series above,
// whose first omitted term there is below 2^-60 of the value.
fn log_factorial(k: i64) -> f64 {
    if k < 128 { return LOG_FACT[k]; }
    let x = (k + 1) as f64;
    let r = 1.0 / x;
    let r2 = r * r;
    let series = r * (1.0 / 12.0 - r2 * (1.0 / 360.0 - r2 * (1.0 / 1260.0 - r2 * (1.0 / 1680.0))));
    return (x - 0.5) * log(x) - x + 0.91893853320467274178 + series;
}

// ---------------------------------------------------------------------------
// Normal and exponential: ziggurats
//
// One 64-bit output picks a layer from its low 8 bits and a point in the
// layer from its high bits; 98.8% (normal) and 98.9% (exponential) of
// points lie under the next layer and return at once. The rest are tested
// against the density, or, in the base layer, go to the exact tail sampler.

// Standard normal: mean 0, variance 1.
fn normal<G>(g: G&) -> f64 {
    loop {
        let r = next_u64(g);
        let i = (r & 255) as! i64;
        let m = r >> 12;                    // 52 bits, independent of i and the sign
        var x = (m as! i64) * ZIG_NORM_W[i];
        if m >= ZIG_NORM_K[i] {
            if i == 0 {
                x = normal_tail(g);
            } else if ZIG_NORM_F[i] + uniform(g) * (ZIG_NORM_F[i + 1] - ZIG_NORM_F[i]) >= exp(-0.5 * x * x) {
                continue;
            }
        }
        return x * SIGN[((r >> 8) & 1) as! i64];
    }
}

// The normal beyond R, exactly (Marsaglia, 1964).
fn normal_tail<G>(g: G&) -> f64 {
    loop {
        let x = -log(uniform_pos(g)) * ZIG_NORM_INV_R;
        let y = -log(uniform_pos(g));
        if y + y > x * x { return ZIG_NORM_R + x; }
    }
}

// Normal with the given mean and standard deviation.
inline fn normal<G>(g: G&, mean: f64, sd: f64) -> f64 { mean + sd * normal(g) }

// Standard exponential: rate 1.
fn exponential<G>(g: G&) -> f64 {
    var base = 0.0;
    loop {
        let r = next_u64(g);
        let i = (r & 255) as! i64;
        let m = r >> 11;                    // 53 bits
        let x = (m as! i64) * ZIG_EXP_W[i];
        if m < ZIG_EXP_K[i] { return base + x; }
        if i == 0 {
            base += ZIG_EXP_R;              // memoryless: the tail is R + another draw
        } else if ZIG_EXP_F[i] + uniform(g) * (ZIG_EXP_F[i + 1] - ZIG_EXP_F[i]) < exp(-x) {
            return base + x;
        }
    }
}

// Exponential with the given rate (mean 1 / rate).
inline fn exponential<G>(g: G&, rate: f64) -> f64 { exponential(g) / rate }

// ---------------------------------------------------------------------------
// Continuous distributions

// Gamma with the given shape (> 0) and scale 1 (Marsaglia and Tsang, 2000).
// A shape below 1 draws shape + 1 and multiplies by U^(1/shape), in logs, so
// that tiny shapes underflow only where the true value does.
fn gamma<G>(g: G&, shape: f64) -> f64 {
    assert(shape >= 0.0);
    if shape == 1.0 { return exponential(g); }
    if shape == 0.0 { return 0.0; }
    if shape < 1.0 {
        // Gamma(shape + 1) times U^(1/shape). pow is one call and correctly
        // rounded to within an ulp; only where it underflows does the product
        // go through logs.
        let y = gamma_mt(g, shape + 1.0);
        let u = uniform_pos(g);
        let p = pow(u, 1.0 / shape);
        return if p > 1.0e-290 { y * p } else { exp(log(y) + log(u) / shape) };
    }
    return gamma_mt(g, shape);
}

// Gamma with the given shape and scale (mean shape * scale).
inline fn gamma<G>(g: G&, shape: f64, scale: f64) -> f64 { gamma(g, shape) * scale }

// Marsaglia and Tsang's method for shape >= 1, with the log test written in
// t = c x so that it keeps its precision when the shape is large:
// 1 - v + log v = 3 log1p(t) - t (3 + t (3 + t)).
fn gamma_mt<G>(g: G&, a: f64) -> f64 {
    let d = a - 1.0 / 3.0;
    let c = 1.0 / sqrt(9.0 * d);
    loop {
        let x = normal(g);
        let t = c * x;
        if t <= -1.0 { continue; }
        let v1 = t * (3.0 + t * (3.0 + t));     // v - 1, v = (1 + t)^3
        let u = uniform(g);
        let x2 = x * x;
        if u < 1.0 - 0.0331 * x2 * x2 { return d * (1.0 + v1); }
        if log(u) < 0.5 * x2 + d * (3.0 * log1p(t) - v1) { return d * (1.0 + v1); }
    }
}

// Beta(a, b), a, b > 0. Both at most 1: Johnk's method, in logs where the
// powers underflow. Otherwise from two gammas.
fn beta<G>(g: G&, a: f64, b: f64) -> f64 {
    assert(a > 0.0 && b > 0.0);
    if a <= 1.0 && b <= 1.0 {
        loop {
            let u = uniform_pos(g);
            let v = uniform_pos(g);
            let x = pow(u, 1.0 / a);
            let y = pow(v, 1.0 / b);
            let s = x + y;
            if s <= 1.0 {
                if s > 0.0 { return x / s; }
                var lx = log(u) / a;
                var ly = log(v) / b;
                let lm = max(lx, ly);
                lx -= lm;
                ly -= lm;
                return exp(lx - log(exp(lx) + exp(ly)));
            }
        }
    }
    let x = gamma(g, a);
    let y = gamma(g, b);
    return x / (x + y);
}

// Chi-squared with k degrees of freedom (> 0).
inline fn chi_squared<G>(g: G&, k: f64) -> f64 { 2.0 * gamma(g, 0.5 * k) }

// Student's t with nu degrees of freedom (> 0).
fn student_t<G>(g: G&, nu: f64) -> f64 {
    let z = normal(g);
    let w = gamma(g, 0.5 * nu);
    return z * sqrt(0.5 * nu / w);
}

// Fisher's F with d1 and d2 degrees of freedom.
fn fisher_f<G>(g: G&, d1: f64, d2: f64) -> f64 {
    let x = gamma(g, 0.5 * d1) / d1;
    let y = gamma(g, 0.5 * d2) / d2;
    return x / y;
}

// Lognormal: exp of a normal with the given mean and standard deviation.
inline fn lognormal<G>(g: G&, mu: f64, sigma: f64) -> f64 { exp(mu + sigma * normal(g)) }

// Cauchy with the given location and scale: a ratio of normals, so both
// tails are exact.
fn cauchy<G>(g: G&, loc: f64, scale: f64) -> f64 {
    let x = normal(g);
    let y = normal(g);
    return loc + scale * (x / y);
}

// Laplace (double exponential) with the given location and scale.
fn laplace<G>(g: G&, loc: f64, scale: f64) -> f64 {
    let e = exponential(g);
    return loc + scale * e * sign(g);
}

// Logistic with the given location and scale: log(E1 / E2), which has the
// logistic distribution and exact tails on both sides.
fn logistic<G>(g: G&, loc: f64, scale: f64) -> f64 {
    let a = exponential(g);
    let b = exponential(g);
    return loc + scale * log(a / b);
}

// Gumbel (extreme value, maximum) with the given location and scale.
inline fn gumbel<G>(g: G&, loc: f64, scale: f64) -> f64 { loc - scale * log(exponential(g)) }

// Weibull with the given shape k and scale.
fn weibull<G>(g: G&, k: f64, scale: f64) -> f64 {
    assert(k > 0.0);
    return scale * pow(exponential(g), 1.0 / k);
}

// Rayleigh with the given scale.
inline fn rayleigh<G>(g: G&, scale: f64) -> f64 { scale * sqrt(2.0 * exponential(g)) }

// Pareto (type I): x >= xm, density alpha xm^alpha / x^(alpha + 1).
inline fn pareto<G>(g: G&, xm: f64, alpha: f64) -> f64 { xm * exp(exponential(g) / alpha) }

// Lomax (Pareto type II), NumPy's `pareto`: x >= 0, the Pareto shifted to 0.
inline fn lomax<G>(g: G&, alpha: f64, scale: f64) -> f64 { scale * expm1(exponential(g) / alpha) }

// A power law: density proportional to x^-alpha on [xmin, xmax], with
// 0 < xmin < xmax (xmax may be infinite when alpha > 1). Inverted in a form
// that keeps its precision as alpha nears 1, and exact at alpha == 1.
fn power_law<G>(g: G&, alpha: f64, xmin: f64, xmax: f64) -> f64 {
    assert(xmin > 0.0 && xmax > xmin);
    if !is_finite(xmax) {
        assert(alpha > 1.0);
        return pareto(g, xmin, alpha - 1.0);
    }
    let u = uniform(g);
    if alpha == 1.0 { return xmin * exp(u * log(xmax / xmin)); }
    let s = 1.0 - alpha;
    let q = expm1(s * log(xmax / xmin));       // (xmax / xmin)^s - 1
    return min(xmax, xmin * exp(log1p(u * q) / s));
}

// Triangular on [left, right] with the given mode.
fn triangular<G>(g: G&, left: f64, mode: f64, right: f64) -> f64 {
    assert(left <= mode && mode <= right && left < right);
    let base = right - left;
    let lb = mode - left;
    let ru = right - mode;
    let u = uniform(g);
    if u <= lb / base { return left + sqrt(u * base * lb); }
    return right - sqrt((1.0 - u) * base * ru);
}

// Wald (inverse Gaussian) with the given mean and scale (Michael, Schucany
// and Haas, 1976).
fn wald<G>(g: G&, mean: f64, scale: f64) -> f64 {
    let mu2l = mean / (2.0 * scale);
    let y0 = normal(g);
    let y = mean * y0 * y0;
    let x = mean + mu2l * (y - sqrt(4.0 * scale * y + y * y));
    if uniform(g) <= mean / (mean + x) { return x; }
    return mean * mean / x;
}

// von Mises on [-pi, pi) about mu with concentration kappa >= 0 (Best and
// Fisher, 1979). Below 1e-8 it is uniform on the circle; above 1e6, the
// wrapped normal it tends to, to double precision.
fn von_mises<G>(g: G&, mu: f64, kappa: f64) -> f64 {
    assert(kappa >= 0.0);
    var x = 0.0;
    if kappa < 1e-8 {
        x = PI * (2.0 * uniform(g) - 1.0);
    } else if kappa > 1e6 {
        x = mu + normal(g) / sqrt(kappa);
    } else {
        // rho = (r - sqrt(2 r)) / (2 kappa); s = (1 + rho^2) / (2 rho).
        var s = 0.0;
        if kappa < 1e-5 {
            s = 1.0 / kappa + kappa;
        } else {
            let r = 1.0 + sqrt(1.0 + 4.0 * kappa * kappa);
            let rho = (r - sqrt(2.0 * r)) / (2.0 * kappa);
            s = (1.0 + rho * rho) / (2.0 * rho);
        }
        var w = 0.0;
        loop {
            let z = cos(PI * uniform(g));
            w = (1.0 + s * z) / (s + z);
            let y = kappa * (s - w);
            let v = uniform_pos(g);
            if y * (2.0 - y) - v >= 0.0 || log(y / v) + 1.0 - y >= 0.0 { break; }
        }
        x = acos(max(-1.0, min(1.0, w)));
        x *= sign(g);
        x += mu;
    }
    // Wrap into [-pi, pi).
    let t = x + PI;
    return t - TAU * floor(t / TAU) - PI;
}

// ---------------------------------------------------------------------------
// Discrete distributions

// Poisson with mean lam >= 0. Below 10, inversion by sequential search
// from 0; from 10, Hormann's transformed rejection (PTRS, 1993), whose
// expected cost does not grow with lam.
//
// The setup that depends only on lam is inline and the sampling loops are
// not, so in a loop with a fixed lam the C compiler can hoist the setup out
// of it, as a distribution object would.
inline fn poisson<G>(g: G&, lam: f64) -> i64 {
    assert(lam >= 0.0 && lam < 1e18);
    if lam >= 10.0 {
        let b = 0.931 + 2.53 * sqrt(lam);
        return poisson_ptrs(g, lam, log(lam), b, -0.059 + 0.02483 * b,
                            1.1239 + 1.1328 / (b - 3.4), 0.9277 - 3.6224 / (b - 2.0));
    }
    if lam == 0.0 { return 0; }
    return poisson_inversion(g, lam, exp(-lam));
}

fn poisson_inversion<G>(g: G&, lam: f64, p0: f64) -> i64 {
    loop {
        var u = uniform(g);
        var p = p0;
        var k = 0;
        while u >= p {
            u -= p;
            k++;
            if k > 120 { break; }   // the rounding of the remainder; at most 1e-100 of the mass
            p *= lam * RECIP[k];
        }
        if k <= 120 { return k; }
    }
}

fn poisson_ptrs<G>(g: G&, lam: f64, loglam: f64, b: f64, a: f64, inv_alpha: f64, vr: f64) -> i64 {
    loop {
        let u = uniform(g) - 0.5;
        let v = uniform_pos(g);
        let us = 0.5 - abs(u);
        // floor(x) as an integer. Below 0 there is no mass; at 2^62 and
        // beyond, far past lam < 1e18, none a double can show. In between, a
        // truncating conversion is the floor in one instruction, where floor
        // itself is a library call on baseline x86-64.
        let x = (2.0 * a / us + b) * u + lam + 0.43;
        if x < 0.0 || x >= 4.6e18 { continue; }
        let k = x as! i64;
        if us >= 0.07 && v <= vr { return k; }
        if us < 0.013 && v > us { continue; }
        if log(v * inv_alpha / (a / (us * us) + b)) <= -lam + k * loglam - log_factorial(k) {
            return k;
        }
    }
}

// Binomial: successes in n trials of probability p. Inversion when
// n min(p, 1 - p) < 10; Hormann's BTRS (1993) above, with exact
// log-factorials in its final test. As for poisson, the inversion's setup is
// inline so that a loop with fixed parameters can hoist it.
inline fn binomial<G>(g: G&, n: i64, p: f64) -> i64 {
    assert(n >= 0 && p >= 0.0 && p <= 1.0);
    if n == 0 || p == 0.0 { return 0; }
    if p == 1.0 { return n; }
    let pl = if p > 0.5 { 1.0 - p } else { p };
    var k = 0;
    if n * pl < 10.0 {
        let np = n * pl;
        let q = 1.0 - pl;
        k = binomial_inversion(g, n, exp(n * log1p(-pl)), pl / q,
                               min(n as f64, np + 10.0 * sqrt(np * q + 1.0)));
    } else {
        k = binomial_btrs(g, n, pl);
    }
    return if p > 0.5 { n - k } else { k };
}

fn binomial_btrs<G>(g: G&, n: i64, p: f64) -> i64 {
    let q = 1.0 - p;
    let spq = sqrt(n * p * q);
    let b = 1.15 + 2.53 * spq;
    let a = -0.0873 + 0.0248 * b + 0.01 * p;
    let c = n * p + 0.5;
    let vr = 0.92 - 4.2 / b;
    let alpha = (2.83 + 5.1 / b) * spq;
    let lpq = log(p / q);
    let m = floor((n + 1) * p) as i64;
    let h = log_factorial(m) + log_factorial(n - m);
    loop {
        let u = uniform(g) - 0.5;
        var v = uniform_pos(g);
        let us = 0.5 - abs(u);
        let x = (2.0 * a / us + b) * u + c;
        if x < 0.0 || x >= n + 1 { continue; }  // floor(x) outside [0, n]
        let k = x as! i64;                      // floor(x), x being >= 0
        if us >= 0.07 && v <= vr { return k; }
        v = log(v * alpha / (a / (us * us) + b));
        if v <= h - log_factorial(k) - log_factorial(n - k) + (k - m) * lpq { return k; }
    }
}

// Sequential search from 0, with the probabilities by recurrence: p0 is
// q^n, r is p / q, and the search gives up past bound (at most 44, n p
// being below 10), where the remaining mass is negligible.
fn binomial_inversion<G>(g: G&, n: i64, p0: f64, r: f64, bound: f64) -> i64 {
    loop {
        var u = uniform(g);
        var px = p0;
        var k = 0;
        while u >= px {
            u -= px;
            k++;
            if k > bound { break; }
            px *= r * (n - k + 1) * RECIP[k];
        }
        if k <= bound { return k; }
    }
}

// Bernoulli trials of probability p up to and including the first success
// (so at least 1).
fn geometric<G>(g: G&, p: f64) -> i64 {
    assert(p > 0.0 && p <= 1.0);
    if p == 1.0 { return 1; }
    if p >= 0.333 {
        let q = 1.0 - p;
        var u = uniform(g);
        var k = 1;
        var pk = p;
        while u >= pk {
            u -= pk;
            pk *= q;
            k++;
        }
        return k;
    }
    let x = ceil(exponential(g) / -log1p(-p));
    return if x >= 9.2e18 { 9223372036854775807 } else { max(1, x as i64) };
}

// Failures before the n-th success in trials of probability p (n > 0 may be
// fractional): Poisson with a gamma-distributed mean.
fn negative_binomial<G>(g: G&, n: f64, p: f64) -> i64 {
    assert(n > 0.0 && p > 0.0 && p <= 1.0);
    if p == 1.0 { return 0; }
    return poisson(g, gamma(g, n) * ((1.0 - p) / p));
}

// Good items among nsample drawn without replacement from ngood good and
// nbad bad ones: inversion from the mode, outwards, with the probabilities
// by recurrence from the mode's, which comes from exact log-factorials. The
// expected cost grows with the standard deviation.
fn hypergeometric<G>(g: G&, ngood: i64, nbad: i64, nsample: i64) -> i64 {
    assert(ngood >= 0 && nbad >= 0 && nsample >= 0 && nsample <= ngood + nbad);
    let total = ngood + nbad;
    let lo = max(0, nsample - nbad);
    let hi = min(nsample, ngood);
    if lo == hi { return lo; }
    // The mode, and its probability.
    let mode = min(hi, max(lo, ((nsample + 1) * (ngood + 1)) / (total + 2)));
    let pm = exp(log_factorial(ngood) - log_factorial(mode) - log_factorial(ngood - mode) +
                 log_factorial(nbad) - log_factorial(nsample - mode) - log_factorial(nbad - nsample + mode) -
                 (log_factorial(total) - log_factorial(nsample) - log_factorial(total - nsample)));
    loop {
        var u = uniform(g) - pm;
        if u < 0.0 { return mode; }
        // Alternate down and up from the mode, taking the larger step first.
        var kd = mode;
        var pd = pm;
        var ku = mode;
        var pu = pm;
        loop {
            var moved = false;
            if kd > lo {
                // p(k - 1) = p(k) k (nbad - nsample + k) / ((ngood - k + 1) (nsample - k + 1))
                pd *= (kd as f64) * (nbad - nsample + kd) / ((ngood - kd + 1) as f64 * (nsample - kd + 1));
                kd--;
                u -= pd;
                if u < 0.0 { return kd; }
                moved = true;
            }
            if ku < hi {
                // p(k + 1) = p(k) (ngood - k) (nsample - k) / ((k + 1) (nbad - nsample + k + 1))
                pu *= ((ngood - ku) as f64) * (nsample - ku) / ((ku + 1) as f64 * (nbad - nsample + ku + 1));
                ku++;
                u -= pu;
                if u < 0.0 { return ku; }
                moved = true;
            }
            if !moved { break; }       // the rounding of the sum: draw again
        }
    }
}

// Zipf: P(k) proportional to k^-a for k >= 1, a > 1 (Devroye's rejection).
fn zipf<G>(g: G&, a: f64) -> i64 {
    assert(a > 1.0);
    let am1 = a - 1.0;
    let b = pow(2.0, am1);
    loop {
        let u = uniform_pos(g);
        let v = uniform(g);
        let x = floor(pow(u, -1.0 / am1));
        if x >= 9.2e18 || x < 1.0 { continue; }
        let t = pow(1.0 + 1.0 / x, am1);
        if v * x * (t - 1.0) / (b - 1.0) <= t / b { return x as i64; }
    }
}

// Logarithmic series with parameter 0 < p < 1 (Kemp, 1981).
fn log_series<G>(g: G&, p: f64) -> i64 {
    assert(p > 0.0 && p < 1.0);
    let r = log1p(-p);
    loop {
        let v = uniform_pos(g);
        if v >= p { return 1; }
        let u = uniform(g);
        let q = -expm1(r * u);
        if v <= q * q {
            let k = floor(1.0 + log(v) / log(q));
            if k < 1.0 || k >= 9.2e18 { continue; }
            return k as i64;
        }
        if v >= q { return 1; }
        return 2;
    }
}

// Counts of n draws among categories of the given probabilities (which
// must sum to 1 or less; the last category takes the rest), into out:
// conditional binomials.
fn multinomial<G>(g: G&, n: i64, probs: f64[:], out: i64[:]) {
    assert(out.len == probs.len && probs.len > 0);
    var left = n;
    var rest = 1.0;
    for p, i in probs {
        if i == probs.len - 1 || left == 0 {
            out[i] = if i == probs.len - 1 { left } else { 0 };
            continue;
        }
        let k = if p >= rest { left } else { binomial(g, left, max(0.0, p / rest)) };
        out[i] = k;
        left -= k;
        rest -= p;
    }
}

// An index drawn with probability proportional to weights[i] (>= 0, not
// all zero): one pass to sum, one to search. For many draws from the same
// weights, an alias_table is O(1) each.
fn weighted_index<G>(g: G&, weights: f64[:]) -> i64 {
    var total = 0.0;
    for w in weights {
        assert(w >= 0.0);
        total += w;
    }
    assert(total > 0.0);
    var u = uniform(g) * total;
    var last = 0;
    for w, i in weights {
        if w > 0.0 {
            if u < w { return i; }
            u -= w;
            last = i;
        }
    }
    return last;        // the rounding of the sum
}

// Walker's alias method, built by Vose's algorithm: O(n) to build from
// weights, then O(1) per draw. Each slot keeps its own index with
// probability `prob`, and gives `alias` otherwise.
struct alias_slot { prob: f64, alias: i64 }
struct alias_table { slots: alias_slot[>..] = [] }

fn alias_table_from(weights: f64[:]) -> alias_table {
    let n = weights.len;
    assert(n > 0);
    var total = 0.0;
    for w in weights {
        assert(w >= 0.0);
        total += w;
    }
    assert(total > 0.0);
    var t = alias_table {};
    var small: i64[>..] = [];
    var large: i64[>..] = [];
    for w, i in weights {
        let p = w * n / total;
        t.slots.push(alias_slot { p, i });
        if p < 1.0 { small.push(i); } else { large.push(i); }
    }
    while small.len > 0 && large.len > 0 {
        let s = small.pop();
        let l = large[large.len - 1];
        t.slots[s].alias = l;
        t.slots[l].prob = (t.slots[l].prob + t.slots[s].prob) - 1.0;
        if t.slots[l].prob < 1.0 {
            large.pop();
            small.push(l);
        }
    }
    // What is left is 1 up to rounding.
    for i in large { t.slots[i].prob = 1.0; }
    for i in small { t.slots[i].prob = 1.0; }
    return t;
}

inline fn sample<G>(g: G&, t) -> i64 {
    let i = below(g, t.slots.len);
    let s = t.slots[i];
    return if uniform(g) < s.prob { i } else { s.alias };
}

// ---------------------------------------------------------------------------
// Directions, points and vectors

// A uniformly random direction in 3-D (Marsaglia, 1972).
fn unit_vector3<G>(g: G&) -> f64, f64, f64 {
    loop {
        let a = 2.0 * uniform(g) - 1.0;
        let b = 2.0 * uniform(g) - 1.0;
        let s = a * a + b * b;
        if s < 1.0 && s > 0.0 {
            let f = 2.0 * sqrt(1.0 - s);
            return a * f, b * f, 1.0 - 2.0 * s;
        }
    }
}

// A uniformly random direction in 2-D.
fn unit_vector2<G>(g: G&) -> f64, f64 {
    let t = TAU * uniform(g);
    return cos(t), sin(t);
}

// A uniformly random direction in out.len dimensions: normals, normalized.
fn unit_vector<G>(g: G&, out: f64[:]) {
    assert(out.len > 0);
    loop {
        var s = 0.0;
        for i in 0..out.len {
            let x = normal(g);
            out[i] = x;
            s += x * x;
        }
        if s > 0.0 {
            let f = 1.0 / sqrt(s);
            for i in 0..out.len { out[i] *= f; }
            return;
        }
    }
}

// A uniformly random point in the unit ball, by rejection from the cube.
fn in_ball3<G>(g: G&) -> f64, f64, f64 {
    loop {
        let x = 2.0 * uniform(g) - 1.0;
        let y = 2.0 * uniform(g) - 1.0;
        let z = 2.0 * uniform(g) - 1.0;
        if x * x + y * y + z * z < 1.0 { return x, y, z; }
    }
}

// A uniformly random point in the unit disk, by rejection from the square.
fn in_disk<G>(g: G&) -> f64, f64 {
    loop {
        let x = 2.0 * uniform(g) - 1.0;
        let y = 2.0 * uniform(g) - 1.0;
        if x * x + y * y < 1.0 { return x, y; }
    }
}

// mean + L z for z standard normal, into out: a multivariate normal whose
// covariance is L L^T. l is the lower-triangular factor (a Cholesky factor),
// row-major n x n; only its lower triangle is read.
fn multivariate_normal<G>(g: G&, mean: f64[:], l: f64[:], out: f64[:]) {
    let n = mean.len;
    assert(out.len == n && l.len == n * n);
    var z: f64[>..] = [];
    z.resize(n, 0.0);
    for i in 0..n { z[i] = normal(g); }
    for i in 0..n {
        var s = mean[i];
        for j in 0..i + 1 { s += l[i * n + j] * z[j]; }
        out[i] = s;
    }
}

// Dirichlet with the given concentrations (> 0), into out. Gamma variates
// in logs, so that small concentrations, whose gammas all underflow, still
// give a point on the simplex.
fn dirichlet<G>(g: G&, alpha: f64[:], out: f64[:]) {
    assert(out.len == alpha.len && alpha.len > 0);
    var top = -1.0 / 0.0;
    for a, i in alpha {
        assert(a > 0.0);
        let lx = if a < 1.0 {
            log(gamma_mt(g, a + 1.0)) + log(uniform_pos(g)) / a
        } else {
            log(gamma_mt(g, a))
        };
        out[i] = lx;
        top = max(top, lx);
    }
    var s = 0.0;
    for i in 0..out.len {
        out[i] = exp(out[i] - top);
        s += out[i];
    }
    for i in 0..out.len { out[i] /= s; }
}

// ---------------------------------------------------------------------------
// Shuffling and sampling

// Fisher-Yates, with exactly uniform indices.
fn shuffle<G, T>(g: G&, xs: T[:]) {
    var h = g;
    var i = xs.len - 1;
    while i > 0 {
        let j = below(h, i + 1);
        swap_at(xs, i, j);
        i -= 1;
    }
    g = h;
}

// A random permutation of 0..n.
fn permutation<G>(g: G&, n: i64) -> i64[>..] {
    var p: i64[>..] = [];
    for i in 0..n { p.push(i); }
    shuffle(g, p);
    return p;
}

// One element of xs, uniformly.
inline fn choice<G, T>(g: G&, xs: T[:]) -> T { xs[below(g, xs.len)] }

// k distinct indices from 0..n, in random order, each k-subset (and each
// order) equally likely. A partial Fisher-Yates over 0..n when k is a good
// part of n; Floyd's algorithm with a set of what is taken otherwise.
fn sample_indices<G>(g: G&, n: i64, k: i64) -> i64[>..] {
    assert(k >= 0 && k <= n);
    var out: i64[>..] = [];
    if k * 4 >= n {
        var p: i64[>..] = [];
        for i in 0..n { p.push(i); }
        for i in 0..k {
            let j = i + below(g, n - i);
            swap_at(p, i, j);
            out.push(p[i]);
        }
        return out;
    }
    var taken = dictionary<i64, bool> {};
    for j in n - k..n {
        let t = below(g, j + 1);
        if taken.insert(t, true) {
            out.push(t);
        } else {
            taken.insert(j, true);
            out.push(j);
        }
    }
    shuffle(g, out);        // Floyd's order is not uniform
    return out;
}

// ---------------------------------------------------------------------------
// Filling arrays

fn fill_u64<G>(g: G&, out: u64[:]) {
    var h = g;
    for i in 0..out.len { out[i] = next_u64(h); }
    g = h;
}

fn fill_uniform<G>(g: G&, out: f64[:]) {
    var h = g;
    for i in 0..out.len { out[i] = uniform(h); }
    g = h;
}

fn fill_normal<G>(g: G&, out: f64[:]) {
    var h = g;
    for i in 0..out.len { out[i] = normal(h); }
    g = h;
}

fn fill_exponential<G>(g: G&, out: f64[:]) {
    var h = g;
    for i in 0..out.len { out[i] = exponential(h); }
    g = h;
}
