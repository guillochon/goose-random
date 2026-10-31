# random

Random number generators and distributions for [Goose](https://aardappel.github.io/goose/)
([aardappel/goose](https://github.com/aardappel/goose)), in pure Goose: no C, no build step,
nothing to install. Fast generators, 30-odd distributions with exact algorithms, shuffling and
sampling, and non-overlapping streams for threads.

```goose
import random from "guillochon/goose-random@main";   // or from "../goose-random" for a local checkout

fn main() {
    var g = random::seed(42);
    let z = random::normal(g);                  // standard normal
    let k = random::poisson(g, 3.5);
    let x, y, w = random::unit_vector3(g);      // a random direction
    var xs: f64[>..] = [];
    xs.resize(1000, 0.0);
    random::fill_normal(g, xs);
    print(z, " ", k, " ", x, " ", y, " ", w, " ", xs[0]);
}
```

Goose fetches the package the first time a program uses it. There is no tagged release yet: pin a
commit (`"guillochon/goose-random@<commit>"`) for a reproducible build, since a branch like
`@main` is pinned at whatever was fetched first, until `goose fetch --update`.

## Conventions

- A **generator** is a small struct you keep in a `var` and pass to every call:
  `random::normal(g)`. `random::seed(s)` gives the default, xoshiro256++, from any 64-bit seed;
  `random::seed_from_entropy()` seeds it from the operating system.
- Every distribution is **generic over the generator**, so it takes any of the generators below,
  std's own `rng`, or one of yours: declare `fn random::next_u64(g: mine&) -> u64` and they all
  work with it.
- Same seed, same numbers, on every platform and compiler: the generators are integer arithmetic,
  and the distributions use only correctly rounded operations and the C library's `exp`, `log`,
  `pow`, `sqrt`, `cos` and `acos` (whose last bit can differ between C libraries).
- Parameters are checked with `assert`: a negative standard deviation, `p > 1` or an empty weight
  list aborts.
- Integer results are `i64`. Functions with several results return them as a tuple:
  `let x, y, z = random::unit_vector3(g)`.
- Everything is in namespace `random`. Goose's UFCS doesn't cross namespaces, so write
  `random::normal(g)`, not `g.normal()`.

## API

| Area | Functions |
|---|---|
| Generators | `xoshiro256pp` (default), `xoshiro256ss`, `sfc64`, `splitmix64`; `next_u64(g)`, `next_u32(g)` |
| Seeding | `seed(s)`, `seed_from_entropy()`, `seed_xoshiro256ss(s)`, `seed_sfc64(s)`, `sfc64_from_words(a, b, c)` (NumPy's SFC64 state), `seed_splitmix64(s)` |
| Streams | `split(g)` (a stream, and g jumps 2^128 ahead), `jump(g)`, `long_jump(g)` (2^192) |
| Uniform | `uniform(g)` [0, 1), `uniform(g, lo, hi)`, `uniform_pos(g)` (0, 1], `uniform_open(g)` (0, 1), `uniform_dense(g)` (every double in (0, 1)), `uniform_f32(g)` |
| Integers | `below(g, n)` [0, n), `below_u64(g, n)`, `between(g, lo, hi)` [lo, hi], `bernoulli(g, p)`, `coin(g)` |
| Normal family | `normal(g)`, `normal(g, mean, sd)`, `lognormal(g, mu, sigma)`, `multivariate_normal(g, mean, l, out)` |
| Gamma family | `exponential(g)`, `exponential(g, rate)`, `gamma(g, shape)`, `gamma(g, shape, scale)`, `beta(g, a, b)`, `chi_squared(g, k)`, `student_t(g, nu)`, `fisher_f(g, d1, d2)`, `dirichlet(g, alpha, out)` |
| Other continuous | `cauchy`, `laplace`, `logistic`, `gumbel` (each `(g, loc, scale)`), `weibull(g, k, scale)`, `rayleigh(g, scale)`, `pareto(g, xm, alpha)`, `lomax(g, alpha, scale)`, `power_law(g, alpha, xmin, xmax)`, `triangular(g, left, mode, right)`, `wald(g, mean, scale)`, `von_mises(g, mu, kappa)` |
| Discrete | `poisson(g, lam)`, `binomial(g, n, p)`, `geometric(g, p)`, `negative_binomial(g, n, p)`, `hypergeometric(g, ngood, nbad, nsample)`, `zipf(g, a)`, `log_series(g, p)`, `multinomial(g, n, probs, out)` |
| Categorical | `weighted_index(g, weights)`; for many draws, `alias_table_from(weights)` then `sample(g, table)`, O(1) each |
| Geometry | `unit_vector3(g)`, `unit_vector2(g)`, `unit_vector(g, out)` (any dimension), `in_ball3(g)`, `in_disk(g)` |
| Sampling | `shuffle(g, xs)`, `permutation(g, n)`, `choice(g, xs)`, `sample_indices(g, n, k)` (without replacement) |
| Arrays | `fill_u64`, `fill_uniform`, `fill_normal`, `fill_exponential` (each `(g, out)`) |
| Helpers | `log1p`, `expm1`, `log_factorial(k)` |

`power_law` is the bounded power law p(x) ∝ x^−α on [x_min, x_max], with `xmax` allowed to be
infinite for α > 1, as for an initial mass function. `pareto` is the classic Pareto (type I) with
support x ≥ x_m. `lomax` is NumPy's `pareto`.

## Generators

| Generator | State | Period | Notes |
|---|---|---|---|
| `xoshiro256pp` | 256 bits | 2^256 − 1 | The default. Blackman and Vigna's xoshiro256++: shifts, rotates and adds, no multiplication; passes BigCrush and PractRand. `jump` and `long_jump` move 2^128 and 2^192 outputs ahead. |
| `xoshiro256ss` | 256 bits | 2^256 − 1 | xoshiro256**, the same engine with a multiplying output scrambler. |
| `sfc64` | 256 bits | ≥ 2^64 | Doty-Humphrey's small fast chaotic generator, with a counter. Its output matches NumPy's `SFC64` from the same state. |
| `splitmix64` | 64 bits | 2^64 | Steele, Lea and Flood. Used to expand seeds. |

`seed(s)` fills xoshiro's state with four splitmix64 outputs, as the xoshiro authors recommend,
so nearby seeds give unrelated streams. For threads, call `split(g)` once per thread from the main
thread and pass each thread its generator (a flat struct, so it crosses into a `thread_fn` by
value). Each stream starts 2^128 outputs after the last, so they never overlap.

## Algorithms and accuracy

- **Uniform doubles** are the top 53 bits of an output times 2^−53, as in NumPy and Rust's `rand`:
  every multiple of 2^−53 in [0, 1) with equal probability. `uniform_pos` and `uniform_open` shift
  that grid so the result is never zero, and the samplers use them wherever they take a log.
  `uniform_dense` reaches every double in (0, 1), each with the probability of the real interval
  that rounds to it (Downey's method): values near zero keep all 53 bits instead of becoming
  multiples of 2^−53.
- **Bounded integers** are exactly uniform: Lemire's multiply-and-reject, which needs one 32-bit
  product for ranges below 2^32 and rarely draws twice. `between` covers the full `i64` range.
- **Normal and exponential** are 256-layer ziggurats. The tables in `tables.goose` come from
  `tools/gen_tables.py`, which solves for the layers in 60-digit arithmetic and rounds each entry
  once, so every table value is the correctly rounded value of its definition. One 64-bit output
  supplies the layer (8 bits), the sign (1 bit) and a 52-bit position (53 bits for the
  exponential). These fields don't overlap, which avoids the correlation of Marsaglia and Tsang's
  original version. About 99% of draws are one output, a table lookup and a compare. The normal's
  tail beyond 3.65 uses Marsaglia's exact tail method. The exponential's tail uses its
  memorylessness: the tail is the radius plus another exponential draw. Neither tail is cut off.
- **Gamma** uses Marsaglia and Tsang's method. The log test is written as
  `3 log1p(t) − t(3 + t(3 + t))`, which keeps its precision for shapes up to 10⁸ and beyond, where
  the textbook `1 − v + log v` cancels. For shapes below 1 it draws shape + 1 and multiplies by
  U^(1/shape), one `pow`, switching to logs where that underflows, so a tiny shape underflows
  only where the true value does. **Beta** uses
  Jöhnk's method in logs when both parameters are at most 1, and two gammas otherwise. **Dirichlet**
  works in log space, so tiny concentrations still give a point on the simplex.
- **Poisson** uses inversion below a mean of 10 and Hörmann's transformed rejection (PTRS) above,
  whose cost doesn't grow with the mean. **Binomial** uses inversion when n·min(p, 1 − p) < 10 and
  Hörmann's BTRS above. Both acceptance tests use exact log-factorials: a correctly rounded table
  below 128 and Stirling's series above, whose error there is below 2^−60.
- **Hypergeometric** is inversion outwards from the mode, with probabilities by recurrence from
  the mode's (exact log-factorials). It's exact, and its cost grows with the standard deviation.
- **Heavy and two-sided tails come from the ziggurats**, not from a log of a 53-bit uniform:
  Cauchy is a ratio of normals, logistic is log(E₁/E₂), and Gumbel, Weibull, Rayleigh, Pareto and
  Lomax are transforms of an exponential. Their tails keep their shape far past the 2^−53
  quantile, where an inversion of `uniform` stops.
- `power_law` inverts its CDF in a form, `x_min · exp(log1p(u q) / (1 − α))` with
  `q = expm1((1 − α) log(x_max / x_min))`, that stays accurate as α approaches 1, and is exact at
  α = 1.
- `log1p` and `expm1` (which Goose's `math` lacks) are Goldberg's and Kahan's corrections, good to
  a few units in the last place.

### Validation

- `test/test_random.goose` (179 checks, about 1 s natively) compares the generators' outputs with
  `tools/ref_vectors.py` (independent implementations, and NumPy's own SFC64) and splitmix64's
  published vector. It checks that every bounded integer is exactly uniform by χ², including
  ranges near 2^63 where `x % n` is biased 2:1, and checks the rejection step against a scripted
  generator. Every distribution is checked against its theoretical moments, by a
  Kolmogorov–Smirnov test where the CDF has a closed form, and, for the discrete ones, by χ²
  against exact probabilities. It also counts the normal's and exponential's tails past their
  ziggurat radius (20 million draws each).
- `tools/validate.py` draws a million values per case (or as many as you ask for) for 55
  parameter sets, and tests each against SciPy's distribution. Continuous cases get
  Kolmogorov–Smirnov and a χ² on 200 equally likely bins plus tail bins down to 10⁻⁶; discrete
  cases get χ² over the support.

## Performance

### Test setup

| | |
|---|---|
| CPU | Intel Core i5-12600K (Alder Lake): 6 performance cores, 3.7 GHz base, 4.9 GHz turbo |
| OS | Windows 11 Home 23H2 (build 22631), High performance power plan |
| Goose | built from aardappel/goose master (commit 2ed1a63d), compiling with clang-cl 22.1.3 (Visual Studio 2026); a plain `goose build` |
| NumPy | 2.2.6 on Python 3.12, `Generator` with its default PCG64, filling arrays with `size=` |
| Rust | rustc 1.88.0, `rand` 0.9.5, `rand_distr` 0.5.1 and `rand_xoshiro` 0.7.0's `Xoshiro256PlusPlus` (Goose's default generator), release build with LTO; "native" adds `-C target-cpu=native` |
| C++ | `<random>` from Visual Studio's standard library with `mt19937_64`, clang-cl `/O2 /arch:AVX2` |
| Pinning | Every run on logical CPU 4, a performance core (`bench/run_pinned.ps1`) |

Each case fills an array of 2^20 values. The time is the best of at least five fills and 0.2 s,
in nanoseconds per value, single-threaded. The fastest time in each row and any within 3% of it
are in bold.

| Case | Goose | NumPy | Rust | Rust, native | C++ |
|---|---|---|---|---|---|
| raw 64-bit output¹ | 0.84 | 2.17 | **0.73** | **0.74** | 1.12 |
| `uniform` | 0.95 | 1.39 | **0.88** | **0.89** | 1.27 |
| `normal` | **1.53** | 6.43 | 1.65 | **1.57** | 6.95 |
| `exponential` | **1.55** | 2.30 | 1.75 | 1.81 | 5.61 |
| `below(g, 1000)` | 1.06 | 2.56 | 0.95 | **0.89** | 1.47 |
| `below(g, 3e18)` | **1.81** | 4.66 | 1.97 | **1.83** | 3.27 |
| `uniform_dense` | **7.04** | — | — | — | — |
| `gamma`, shape 0.5 | **6.19** | 33.7 | 8.81 | 7.91 | 42.3 |
| `gamma`, shape 3 | 4.79 | 13.1 | 6.54 | **4.55** | 13.8 |
| `beta(2, 5)` | **9.84** | 28.4 | 56.2 | 52.2 | — |
| `beta(0.5, 0.5)` | **6.11** | 47.6 | 62.7 | 55.0 | — |
| `chi_squared(5)` | 4.97 | 14.2 | 5.00 | **4.78** | 77.0 |
| `student_t(5)` | **6.46** | 23.3 | 6.82 | **6.34** | 21.0 |
| `lognormal` | **4.59** | 11.8 | 9.38 | 8.88 | 20.6 |
| `cauchy` | **2.84** | 16.8 | 11.6 | 10.8 | 12.4 |
| `weibull(1.5)` | **9.14** | 17.6 | 57.7 | 52.7 | 16.7 |
| `von_mises(κ = 4)` | **31.4** | 48.3 | — | — | — |
| `poisson(3)`² | 14.1 | 23.3 | **12.7** | **12.5** | 22.7 |
| `poisson(1000)` | 10.9 | 21.3 | 9.47 | **8.95** | 87.6 |
| `binomial(20, 0.3)`² | 26.1 | 34.2 | **18.1** | **17.7** | 57.5 |
| `binomial(1000, 0.3)` | **14.6** | 29.4 | 21.4 | 19.9 | 150.2 |
| `geometric(0.01)` | **2.60** | 11.8 | 56.6 | 51.3 | 5.74 |
| `hypergeometric(500, 500, 200)` | **29.6** | 68.5 | 39.7 | 36.5 | — |
| alias table, 6 outcomes | 4.70 | 13.4 | 4.92 | **4.48** | 12.6 |
| `shuffle`, per element | 2.45 | 12.1 | 6.44 | **2.34** | 4.93 |

¹ Each library's default generator: xoshiro256++ for Goose and Rust, PCG64 for NumPy, and
mt19937_64 for C++. Goose's xoshiro256** takes 0.75 ns and sfc64 0.98 ns. NumPy's SFC64 is
between 0 and 25% faster than its PCG64 in these cases.

² Goose's samplers take their parameters on every call, so each draw pays for the setup that
depends on them: for these two, an `exp`, a `log` and a `sqrt`. Rust builds a distribution object
once (`Binomial::new(20, 0.3)`) and keeps that setup out of the timed loop.

The big gaps against `rand_distr` and NumPy come from algorithms, not code generation. Goose's
Cauchy is a ratio of two ziggurat normals rather than a `tan`, its Weibull and geometric are
transforms of a ziggurat exponential, its beta for small parameters is Jöhnk's method, and its
gamma for shape below 1 uses a single `pow`.

`/arch:AVX2` doesn't help overall. It halves splitmix64 (0.42 ns), and most cases move by under
5%, but the samplers that lean on the C library's `pow`, `log` and `cos` get slower: Weibull
22 ns, von Mises 48 ns, `binomial(1000, 0.3)` 19.5 ns, `poisson(1000)` 12.6 ns, and the normal
1.73 ns. The plain build is the one to use.
