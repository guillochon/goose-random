// Tests for the random package.
//
//   goose test/test_random.goose                                 (in-process)
//   goose build test/test_random.goose && test/test_random        (native)
//
// The generators against reference outputs (tools/ref_vectors.py, from
// independent implementations and NumPy's SFC64); the uniform and integer
// primitives' ranges and exact uniformity; then every distribution: its
// moments against theory, its CDF by Kolmogorov-Smirnov where it has a
// closed form, and the discrete ones' probabilities by chi-squared against
// exact probabilities. Every draw comes from fixed seeds, so a run is
// deterministic; the statistical bounds are wide enough (about 5 standard
// errors) that a correct sampler passes them, and a wrong one, which is
// usually off by much more, fails.
//
// tools/validate.py runs the larger goodness-of-fit tests with SciPy.

import std;
import math;
import random from "..";

var passed = 0;
var failed = 0;
var group_name: u8[>..] = [];

fn group(name: const u8[:]) {
    group_name.clear();
    group_name.append(name);
}

fn check(ok: bool, what: const u8[:]) {
    if ok { passed++; return; }
    failed++;
    print("FAIL [", group_name, "] ", what);
}

fn check_eq(got: u64[:], want: u64[:], what: const u8[:]) {
    var ok = got.len == want.len;
    if ok { for w, i in want { if got[i] != w { ok = false; } } }
    if ok { passed++; return; }
    failed++;
    print("FAIL [", group_name, "] ", what, ": got ", got, ", want ", want);
}

// An array literal of u64 constants: its elements adapt to u64 here.
fn u64s(xs: u64[:]) -> u64[:] { xs }

fn first<G>(g: G&, n: i64) -> u64[>..] {
    var out: u64[>..] = [];
    for i in 0..n { out.push(random::next_u64(g)); }
    return out;
}

// n draws of F().
fn draw<F>(n: i64) -> f64[>..] {
    var xs: f64[>..] = [];
    for i in 0..n { xs.push(F()); }
    return xs;
}

fn idraw<F>(n: i64) -> i64[>..] {
    var xs: i64[>..] = [];
    for i in 0..n { xs.push(F()); }
    return xs;
}

let N = 200000;
let SE = 5.0;       // standard errors allowed

// The sample mean within SE standard errors of mean, and the sample
// variance within SE of var, both errors estimated from the sample (its
// variance, and its fourth central moment for the variance's).
fn check_moments(xs: f64[:], mean: f64, var_: f64, what: const u8[:]) {
    let n = xs.len as f64;
    var m = 0.0;
    for x in xs { m += x; }
    m /= n;
    var m2 = 0.0;
    var m4 = 0.0;
    for x in xs {
        let d = x - m;
        m2 += d * d;
        m4 += d * d * d * d;
    }
    m2 /= n;
    m4 /= n;
    let se_m = sqrt(m2 / n);
    let se_v = sqrt(max(m4 - m2 * m2, 0.0) / n);
    let ok_m = abs(m - mean) <= SE * se_m + 1e-12 * abs(mean);
    let ok_v = var_ < 0.0 || abs(m2 - var_) <= SE * se_v + 1e-12 * var_;
    if ok_m && ok_v { passed++; return; }
    failed++;
    print("FAIL [", group_name, "] ", what, ": mean ", m, " (want ", mean, " +- ", SE * se_m,
          "), variance ", m2, " (want ", var_, " +- ", SE * se_v, ")");
}

// Kolmogorov-Smirnov against the CDF F: sqrt(n) D below 1.95 (p = 0.001).
fn check_ks<F>(xs: f64[:], what: const u8[:] = "KS") {
    var s: f64[>..] = [];
    s.append(xs);
    sort(s);
    let n = s.len as f64;
    var d = 0.0;
    for x, i in s {
        let f = F(x);
        d = max(d, max(abs(f - i / n), abs((i + 1) / n - f)));
    }
    let stat = sqrt(n) * d;
    if stat < 1.95 { passed++; return; }
    failed++;
    print("FAIL [", group_name, "] ", what, ": KS sqrt(n) D = ", stat);
}

// Chi-squared of the counts of ks against P(k), on [lo, hi]: cells whose
// expected count is under 20 are merged with the next, the mass outside the
// range is a cell of its own, and the statistic must be below the
// Wilson-Hilferty approximation of the 1 - 1e-6 quantile.
fn check_pmf<P>(ks: i64[:], lo: i64, hi: i64, what: const u8[:]) {
    let n = ks.len as f64;
    var counts: i64[>..] = [];
    counts.resize(hi - lo + 1, 0);
    var outside = 0;
    for k in ks {
        if k >= lo && k <= hi { counts[k - lo]++; } else { outside++; }
    }
    var chi = 0.0;
    var df = -1;
    var obs = 0.0;
    var exp_ = 0.0;
    var inside = 0.0;
    for k in lo..hi + 1 {
        let p = P(k);
        inside += p;
        obs += counts[k - lo];
        exp_ += p * n;
        if exp_ >= 20.0 {
            chi += (obs - exp_) * (obs - exp_) / exp_;
            df++;
            obs = 0.0;
            exp_ = 0.0;
        }
    }
    obs += outside;
    exp_ += max(0.0, 1.0 - inside) * n;
    if exp_ >= 1.0 {
        chi += (obs - exp_) * (obs - exp_) / exp_;
        df++;
    } else if obs > 3.0 {
        chi += 1e9;     // draws where there should be none
    }
    if df < 1 {
        // One cell: the draws all fall where the probability is.
        if chi < 1.0 { passed++; return; }
        failed++;
        print("FAIL [", group_name, "] ", what, ": draws outside the support");
        return;
    }
    let z = 4.75;
    let w = 2.0 / (9.0 * df);
    let crit = df * pow(1.0 - w + z * sqrt(w), 3.0);
    if chi < crit { passed++; return; }
    failed++;
    print("FAIL [", group_name, "] ", what, ": chi-squared ", chi, " on ", df, " df, above ", crit);
}

fn log_choose(n: i64, k: i64) -> f64 {
    random::log_factorial(n) - random::log_factorial(k) - random::log_factorial(n - k)
}

// ---------------------------------------------------------------------------

fn test_generators() {
    group("generators");
    var sm = random::seed_splitmix64(1234567);
    check_eq(first(sm, 5), u64s([6457827717110365317 as u64, 3203168211198807973, 9817491932198370423, 4593380528125082431, 16408922859458223821]), "splitmix64");

    var a = random::xoshiro256pp { 1, 2, 3, 4 };
    check_eq(first(a, 6), u64s([41943041 as u64, 58720359, 3588806011781223, 3591011842654386, 9228616714210784205, 9973669472204895162]), "xoshiro256++");
    var b = random::seed(42);
    check_eq(first(b, 4), u64s([15021278609987233951 as u64, 5881210131331364753, 18149643915985481100, 12933668939759105464]), "xoshiro256++ seed(42)");
    a = random::xoshiro256pp { 1, 2, 3, 4 };
    random::jump(a);
    check_eq(first(a, 3), u64s([17043750140134683703 as u64, 2364973248208838314, 13951431646535487319]), "xoshiro256++ jump");
    a = random::xoshiro256pp { 1, 2, 3, 4 };
    random::long_jump(a);
    check_eq(first(a, 3), u64s([13097851138432240629 as u64, 5869259491745178931, 2145365994275058833]), "xoshiro256++ long_jump");

    var c = random::xoshiro256ss { 1, 2, 3, 4 };
    check_eq(first(c, 6), u64s([11520 as u64, 0, 1509978240, 1215971899390074240, 1216172134540287360, 607988272756665600]), "xoshiro256**");
    var d = random::seed_xoshiro256ss(42);
    check_eq(first(d, 4), u64s([1546998764402558742 as u64, 6990951692964543102, 12544586762248559009, 17057574109182124193]), "xoshiro256** seed(42)");
    c = random::xoshiro256ss { 1, 2, 3, 4 };
    random::jump(c);
    check_eq(first(c, 3), u64s([13534147089533256664 as u64, 7126240192422241655, 3805973808039778091]), "xoshiro256** jump");
    c = random::xoshiro256ss { 1, 2, 3, 4 };
    random::long_jump(c);
    check_eq(first(c, 3), u64s([5942309088398569549 as u64, 15625447729937358436, 6925613901769781251]), "xoshiro256** long_jump");

    var e = random::sfc64_from_words(0x0123456789ABCDEF, 0xFEDCBA9876543210, 0x0F1E2D3C4B5A6978);
    check_eq(first(e, 6), u64s([7618646694457798361 as u64, 12251097937939290445, 6299304194219842280, 1949877253283061998, 6301430946280420188, 9590118589597490137]), "sfc64 (as NumPy)");

    // split hands out the state, then jumps.
    var p = random::xoshiro256pp { 1, 2, 3, 4 };
    var q = random::split(p);
    check_eq(first(q, 2), u64s([41943041 as u64, 58720359]), "split returns the stream");
    check_eq(first(p, 1), u64s([17043750140134683703]), "split jumps the source");

    // Same seed, same stream; std's rng plugs in.
    var r1 = random::seed(7);
    var r2 = random::seed(7);
    check(random::normal(r1) == random::normal(r2), "deterministic");
    var s = rng { 5 };
    var t = rng { 5 };
    check(random::uniform(s) == rand_flt(t), "std rng");
}

fn test_uniform() {
    group("uniform");
    var g = random::seed(1);
    var ok = true;
    var lo = 1.0;
    var hi = 0.0;
    var sum = 0.0;
    for i in 0..N {
        let u = random::uniform(g);
        let p = random::uniform_pos(g);
        let o = random::uniform_open(g);
        let f = random::uniform_f32(g);
        let d = random::uniform_dense(g);
        if !(u >= 0.0 && u < 1.0 && p > 0.0 && p <= 1.0 && o > 0.0 && o < 1.0 &&
             f >= 0.0 && f < 1.0 && d > 0.0 && d < 1.0) { ok = false; }
        lo = min(lo, d);
        hi = max(hi, u);
        sum += d;
    }
    check(ok, "ranges");
    check(abs(sum / N - 0.5) < 5.0 * sqrt(1.0 / 12.0 / N), "dense mean");
    check(hi > 0.9999, "reaches near 1");

    // uniform_dense below 2^-53, where the 53-bit forms have nothing: a
    // value under 2^-k with probability 2^-k, here k = 4 and 8 checked by
    // counts, and fine detail: dense values below 2^-60 are not multiples of
    // 2^-53.
    var c4 = 0;
    var c8 = 0;
    for i in 0..N {
        let d = random::uniform_dense(g);
        if d < 0.0625 { c4++; }
        if d < 0.00390625 { c8++; }
    }
    check(abs(c4 - N / 16.0) < 5.0 * sqrt(N / 16.0), "dense below 2^-4");
    check(abs(c8 - N / 256.0) < 5.0 * sqrt(N / 256.0), "dense below 2^-8");

    let xs = draw(N) { random::uniform(g, -2.0, 3.0) };
    check_moments(xs, 0.5, 25.0 / 12.0, "uniform(-2, 3)");
    check_ks(xs) { x => (x + 2.0) / 5.0 };
}

// A generator that returns a fixed sequence, for exact tests of the
// rejection steps.
struct fixed_gen { vals: u64[8], i: i64 }
fn random::next_u64(g: fixed_gen&) -> u64 {
    let v = g.vals[g.i % 8];
    g.i++;
    return v;
}

fn test_integers() {
    group("integers");
    var g = random::seed(2);
    // below: in range, and exactly uniform by chi-squared, for small n, for
    // n just over 2^32, and for n near 2^63, where % would be badly biased.
    for n in [1, 2, 3, 7, 10, 1000] {
        let ks = idraw(N) { random::below(g, n) };
        var ok = true;
        for k in ks { if k < 0 || k >= n { ok = false; } }
        check(ok, "below in range");
        check_pmf(ks, 0, n - 1, "below(n) uniform") { k => 1.0 / n };
    }
    let big = 4294967311;       // 2^32 + 15
    let ks = idraw(N) { random::below(g, big) % 8 };
    check_pmf(ks, 0, 7, "below(2^32 + 15) mod 8") { k => if k < 7 { 536870914.0 / big } else { 536870913.0 / big } };
    // n = 3 * 2^61: x % n would give the low third twice the probability.
    let n3 = 6917529027641081856;
    let thirds = idraw(N) { random::below(g, n3) / 2305843009213693952 };
    check_pmf(thirds, 0, 2, "below(3 * 2^61) by thirds") { k => 1.0 / 3.0 };

    // The rejection: with n = 3 and a 32-bit word whose product lands in the
    // biased low values (2^32 mod 3 = 1, so only a low word of 0 rejects).
    var f = fixed_gen { [0, 0x5555555600000000, 0, 0, 0, 0, 0, 0], 0 };
    check(random::below(f, 3) == 1 && f.i == 2, "Lemire rejects the biased word");

    // between, including the extremes of i64.
    var ok = true;
    for i in 0..1000 {
        let v = random::between(g, -3, 3);
        if v < -3 || v > 3 { ok = false; }
        let w = random::between(g, -9223372036854775807 - 1, 9223372036854775807);
        let z = random::between(g, 9223372036854775806, 9223372036854775807);
        if z < 9223372036854775806 { ok = false; }
    }
    check(ok, "between in range");
    check(random::between(g, 5, 5) == 5, "between one value");
    let bs = idraw(N) { random::between(g, -3, 3) };
    check_pmf(bs, -3, 3, "between uniform") { k => 1.0 / 7.0 };

    let coins = idraw(N) { if random::coin(g) { 1 } else { 0 } };
    check_pmf(coins, 0, 1, "coin") { k => 0.5 };
    let bern = idraw(N) { if random::bernoulli(g, 0.3) { 1 } else { 0 } };
    check_pmf(bern, 0, 1, "bernoulli(0.3)") { k => if k == 1 { 0.3 } else { 0.7 } };
}

// The normal CDF, through erfc (W. J. Cody's rational approximations would
// be more accurate; this one, from Numerical Recipes, is good to 1.2e-7,
// far below what KS resolves at this n).
fn erfc_approx(x: f64) -> f64 {
    let z = abs(x);
    let t = 1.0 / (1.0 + 0.5 * z);
    let r = t * exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 +
        t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 +
            t * (-0.82215223 + t * 0.17087277)))))))));
    return if x >= 0.0 { r } else { 2.0 - r };
}
fn normal_cdf(x: f64) -> f64 { 0.5 * erfc_approx(-x * 0.7071067811865476) }

fn test_normal_exponential() {
    group("normal");
    var g = random::seed(3);
    let xs = draw(N) { random::normal(g) };
    check_moments(xs, 0.0, 1.0, "standard");
    check_ks(xs) { x => normal_cdf(x) };
    // Fourth moment 3 and symmetry: the ziggurat's layers and sign.
    var m4 = 0.0;
    var pos = 0;
    for x in xs {
        m4 += x * x * x * x;
        if x > 0.0 { pos++; }
    }
    check(abs(m4 / N - 3.0) < 5.0 * sqrt(96.0 / N), "fourth moment");
    check(abs(pos - N / 2.0) < 5.0 * sqrt(N / 4.0), "symmetric");
    // The tail beyond R = 3.654 (layer 0 past its rectangle) and beyond 4.5:
    // P(|Z| > 3.654) = 2.581e-4, P(|Z| > 4.5) = 6.795e-6.
    let m = 20000000;
    var t1 = 0;
    var t2 = 0;
    for i in 0..m {
        let z = abs(random::normal(g));
        if z > 3.6541528853610088 { t1++; }
        if z > 4.5 { t2++; }
    }
    check(abs(t1 - m * 2.5812e-4) < 5.0 * sqrt(m * 2.5812e-4), "tail past R");
    check(abs(t2 - m * 6.7953e-6) < 5.0 * sqrt(m * 6.7953e-6), "tail past 4.5");
    let ys = draw(N) { random::normal(g, 10.0, 3.0) };
    check_moments(ys, 10.0, 9.0, "mean 10, sd 3");

    group("exponential");
    let es = draw(N) { random::exponential(g) };
    check_moments(es, 1.0, 1.0, "rate 1");
    check_ks(es) { x => -expm1(-x) };
    var e1 = 0;
    var e2 = 0;
    for i in 0..m {
        let e = random::exponential(g);
        if e > 7.69711747013105 { e1++; }
        if e > 12.0 { e2++; }
    }
    check(abs(e1 - m * exp(-7.69711747013105)) < 5.0 * sqrt(m * exp(-7.69711747013105)), "tail past R");
    check(abs(e2 - m * exp(-12.0)) < 5.0 * sqrt(m * exp(-12.0)) + 1.0, "tail past 12");
    let fs = draw(N) { random::exponential(g, 4.0) };
    check_moments(fs, 0.25, 0.0625, "rate 4");
}

fn expm1(x: f64) -> f64 { random::expm1(x) }

fn test_continuous() {
    var g = random::seed(4);

    group("gamma");
    for k in [0.05, 0.3, 1.0, 1.5, 4.0, 30.0, 1e4] {
        let xs = draw(N) { random::gamma(g, k) };
        check_moments(xs, k, k, "shape");
    }
    let gs = draw(N) { random::gamma(g, 2.0, 3.0) };
    check_moments(gs, 6.0, 18.0, "shape 2 scale 3");
    // Shape 2: CDF 1 - (1 + x) e^-x.
    let g2 = draw(N) { random::gamma(g, 2.0) };
    check_ks(g2) { x => 1.0 - (1.0 + x) * exp(-x) };
    // Tiny shapes underflow to 0 but never give NaN or a negative.
    var ok = true;
    for i in 0..1000 {
        let x = random::gamma(g, 1e-300);
        if !(x >= 0.0) { ok = false; }
    }
    check(ok, "tiny shape");

    group("beta");
    for ab in [[0.1, 0.2], [0.5, 0.5], [1.0, 1.0], [2.0, 5.0], [0.7, 3.0], [50.0, 20.0]] {
        let a = ab[0];
        let b = ab[1];
        let xs = draw(N) { random::beta(g, a, b) };
        let s = a + b;
        check_moments(xs, a / s, a * b / (s * s * (s + 1.0)), "a, b");
    }
    let bs = draw(N) { random::beta(g, 2.0, 1.0) };
    check_ks(bs) { x => x * x };
    ok = true;
    for i in 0..1000 {
        let x = random::beta(g, 1e-3, 1e-3);
        if !(x >= 0.0 && x <= 1.0) { ok = false; }
    }
    check(ok, "tiny parameters stay in [0, 1]");

    group("chi-squared, t, F");
    let cs = draw(N) { random::chi_squared(g, 3.0) };
    check_moments(cs, 3.0, 6.0, "chi-squared(3)");
    let ts = draw(N) { random::student_t(g, 10.0) };
    check_moments(ts, 0.0, 10.0 / 8.0, "t(10)");
    // t with 1 degree of freedom is Cauchy; with 2, CDF 1/2 + x / (2 sqrt(2 + x^2)).
    let t2 = draw(N) { random::student_t(g, 2.0) };
    check_ks(t2) { x => 0.5 + x / (2.0 * sqrt(2.0 + x * x)) };
    let fs = draw(N) { random::fisher_f(g, 5.0, 20.0) };
    check_moments(fs, 20.0 / 18.0, 2.0 * 400.0 * 23.0 / (5.0 * 18.0 * 18.0 * 16.0), "F(5, 20)");

    group("lognormal");
    let ls = draw(N) { random::lognormal(g, 0.5, 0.4) };
    let lm = exp(0.5 + 0.08);
    check_moments(ls, lm, (exp(0.16) - 1.0) * lm * lm, "mu 0.5 sigma 0.4");

    group("cauchy, laplace, logistic, gumbel");
    let cy = draw(N) { random::cauchy(g, 1.0, 2.0) };
    check_ks(cy) { x => 0.5 + atan((x - 1.0) / 2.0) / PI };
    let lp = draw(N) { random::laplace(g, 1.0, 2.0) };
    check_moments(lp, 1.0, 8.0, "laplace");
    check_ks(lp) { x => if x < 1.0 { 0.5 * exp((x - 1.0) / 2.0) } else { 1.0 - 0.5 * exp(-(x - 1.0) / 2.0) } };
    let lg = draw(N) { random::logistic(g, 1.0, 2.0) };
    check_moments(lg, 1.0, PI * PI * 4.0 / 3.0, "logistic");
    check_ks(lg) { x => 1.0 / (1.0 + exp(-(x - 1.0) / 2.0)) };
    let gb = draw(N) { random::gumbel(g, 1.0, 2.0) };
    check_moments(gb, 1.0 + 2.0 * 0.5772156649015329, PI * PI * 4.0 / 6.0, "gumbel");
    check_ks(gb) { x => exp(-exp(-(x - 1.0) / 2.0)) };

    group("weibull, rayleigh, pareto, lomax");
    let wb = draw(N) { random::weibull(g, 1.5, 2.0) };
    check_ks(wb) { x => 1.0 - exp(-pow(x / 2.0, 1.5)) };
    let ry = draw(N) { random::rayleigh(g, 2.0) };
    check_ks(ry) { x => 1.0 - exp(-x * x / 8.0) };
    let pa = draw(N) { random::pareto(g, 2.0, 3.0) };
    check_ks(pa) { x => 1.0 - pow(2.0 / x, 3.0) };
    check_moments(pa, 3.0, 3.0, "pareto(2, 3)");
    let lx = draw(N) { random::lomax(g, 3.0, 2.0) };
    check_ks(lx) { x => 1.0 - pow(1.0 + x / 2.0, -3.0) };

    group("power law");
    for alpha in [-1.0, 0.5, 0.999999, 1.0, 1.000001, 2.35] {
        let xs = draw(N) { random::power_law(g, alpha, 0.1, 100.0) };
        ok = true;
        for x in xs { if x < 0.1 || x > 100.0 { ok = false; } }
        check(ok, "in range");
        let s = 1.0 - alpha;
        check_ks(xs) { x => if alpha == 1.0 { log(x / 0.1) / log(1000.0) } else {
            (pow(x, s) - pow(0.1, s)) / (pow(100.0, s) - pow(0.1, s)) } };
    }
    let inf = draw(N) { random::power_law(g, 2.35, 0.5, 1.0 / 0.0) };
    check_ks(inf) { x => 1.0 - pow(x / 0.5, -1.35) };

    group("triangular, wald, von mises");
    let tr = draw(N) { random::triangular(g, 0.0, 1.0, 4.0) };
    check_moments(tr, 5.0 / 3.0, (16.0 + 1.0 - 4.0) / 18.0, "triangular");
    check_ks(tr) { x => if x <= 1.0 { x * x / 4.0 } else { 1.0 - (4.0 - x) * (4.0 - x) / 12.0 } };
    let wd = draw(N) { random::wald(g, 2.0, 3.0) };
    check_moments(wd, 2.0, 8.0 / 3.0, "wald(2, 3)");
    for kappa in [0.0, 1e-6, 0.5, 4.0, 1e7] {
        let vs = draw(N) { random::von_mises(g, 1.0, kappa) };
        ok = true;
        var c = 0.0;
        var s = 0.0;
        for v in vs {
            if v < -PI || v >= PI { ok = false; }
            c += cos(v - 1.0);
            s += sin(v - 1.0);
        }
        check(ok, "in [-pi, pi)");
        // E cos(X - mu) = I1(kappa) / I0(kappa): 0 at 0, 0.2425 at 0.5,
        // 0.8635 at 4; E sin(X - mu) = 0.
        let want = if kappa == 0.5 { 0.24249997 } else if kappa == 4.0 { 0.86352
        } else if kappa > 1e6 { 1.0 - 0.5 / kappa } else { 0.5 * kappa };
        check(abs(c / N - want) < 5.0 / sqrt(2.0 * N) && abs(s / N) < 5.0 / sqrt(2.0 * N), "mean resultant");
    }
}

fn poisson_pmf(lam: f64, k: i64) -> f64 { exp(-lam + k * log(lam) - random::log_factorial(k)) }

fn test_discrete() {
    var g = random::seed(5);

    group("poisson");
    for lam in [0.0, 0.01, 0.7, 3.0, 9.99, 10.0, 25.0, 300.0, 1e5] {
        let ks = idraw(N) { random::poisson(g, lam) };
        let sd = sqrt(lam);
        let lo = max(0, (lam - 7.0 * sd - 1.0) as i64);
        let hi = (lam + 8.0 * sd + 4.0) as i64;
        if lam == 0.0 {
            var ok = true;
            for k in ks { if k != 0 { ok = false; } }
            check(ok, "lam 0");
            continue;
        }
        check_pmf(ks, lo, hi, "pmf") { k => poisson_pmf(lam, k) };
    }
    let big = idraw(N) { random::poisson(g, 1e12) };
    var xs: f64[>..] = [];
    for k in big { xs.push(k); }
    check_moments(xs, 1e12, 1e12, "lam 1e12");

    group("binomial");
    for np in [[10.0, 0.5], [1.0, 0.3], [20.0, 0.05], [100.0, 0.3], [100.0, 0.97], [1000.0, 0.5], [50.0, 0.2], [5000.0, 0.01]] {
        let n = np[0] as i64;
        let p = np[1];
        let ks = idraw(N) { random::binomial(g, n, p) };
        check_pmf(ks, 0, n, "pmf") { k => exp(log_choose(n, k) + k * log(p) + (n - k) * random::log1p(-p)) };
    }
    check(random::binomial(g, 0, 0.5) == 0 && random::binomial(g, 7, 0.0) == 0 &&
          random::binomial(g, 7, 1.0) == 7, "edges");
    let bb = idraw(N) { random::binomial(g, 1000000000000, 0.25) };
    var bx: f64[>..] = [];
    for k in bb { bx.push(k); }
    check_moments(bx, 2.5e11, 1.875e11, "n 1e12");

    group("geometric, negative binomial");
    for p in [1.0, 0.9, 0.4, 0.333, 0.1, 0.001] {
        let ks = idraw(N) { random::geometric(g, p) };
        let hi = if p == 1.0 { 1 } else { (40.0 / -random::log1p(-p)) as i64 };
        check_pmf(ks, 1, hi, "geometric pmf") { k => p * pow(1.0 - p, (k - 1) as f64) };
    }
    for np in [[3.0, 0.5], [0.5, 0.2], [10.0, 0.9]] {
        let n = np[0];
        let p = np[1];
        let ks = idraw(N) { random::negative_binomial(g, n, p) };
        check_pmf(ks, 0, 400, "negative binomial pmf") { k => exp(
            random::log_factorial(k) * -1.0 + log_gamma_ratio(n, k) + n * log(p) + k * random::log1p(-p)) };
    }

    group("hypergeometric");
    for c in [[10, 5, 7], [5, 10, 7], [50, 50, 20], [1000, 2000, 500], [1, 999, 300], [100000, 100000, 100000], [3, 3, 6]] {
        let good = c[0];
        let bad = c[1];
        let s = c[2];
        let ks = idraw(N) { random::hypergeometric(g, good, bad, s) };
        check_pmf(ks, max(0, s - bad), min(s, good), "pmf") { k =>
            exp(log_choose(good, k) + log_choose(bad, s - k) - log_choose(good + bad, s)) };
    }

    group("zipf, log series");
    let zs = idraw(N) { random::zipf(g, 2.5) };
    // zeta(2.5) = 1.3414872572509171798
    check_pmf(zs, 1, 2000, "zipf(2.5) pmf") { k => pow(k as f64, -2.5) / 1.3414872572509171798 };
    let ls = idraw(N) { random::log_series(g, 0.8) };
    check_pmf(ls, 1, 100, "log series(0.8) pmf") { k => -pow(0.8, k as f64) / (k * random::log1p(-0.8)) };
}

// log(Gamma(n + k) / Gamma(n)) for real n > 0, integer k >= 0, by the sum of
// logs: exact enough for the small k the test uses.
fn log_gamma_ratio(n: f64, k: i64) -> f64 {
    var s = 0.0;
    for i in 0..k { s += log(n + i); }
    return s;
}

fn test_multivariate() {
    var g = random::seed(6);

    group("multinomial");
    var out: i64[4] = [0, 0, 0, 0];
    let probs = [0.1, 0.2, 0.3, 0.4];
    var tot: f64[4] = [0.0, 0.0, 0.0, 0.0];
    var ok = true;
    for i in 0..20000 {
        random::multinomial(g, 50, probs, out);
        var s = 0;
        for c, j in out {
            s += c;
            tot[j] += c;
        }
        if s != 50 { ok = false; }
    }
    check(ok, "counts sum to n");
    for p, j in probs {
        check(abs(tot[j] / 20000.0 - 50.0 * p) < 5.0 * sqrt(50.0 * p * (1.0 - p) / 20000.0), "means");
    }

    group("categorical");
    let w = [1.0, 0.0, 3.0, 6.0];
    let t = random::alias_table_from(w);
    let ks = idraw(N) { random::sample(g, t) };
    check_pmf(ks, 0, 3, "alias table") { k => w[k] / 10.0 };
    let ws = idraw(N) { random::weighted_index(g, w) };
    check_pmf(ws, 0, 3, "weighted_index") { k => w[k] / 10.0 };
    let w2 = [0.5, 0.25, 0.125, 0.0625, 0.0625, 0.0];
    let t2 = random::alias_table_from(w2);
    let k2 = idraw(N) { random::sample(g, t2) };
    check_pmf(k2, 0, 5, "alias table, uneven") { k => w2[k] };

    group("directions");
    var mx = 0.0;
    var mz = 0.0;
    var mzz = 0.0;
    ok = true;
    for i in 0..N {
        let x, y, z = random::unit_vector3(g);
        if abs(x * x + y * y + z * z - 1.0) > 1e-15 { ok = false; }
        mx += x;
        mz += z;
        mzz += z * z;
    }
    check(ok, "unit_vector3 has length 1");
    // Uniform on the sphere: z uniform on [-1, 1].
    check(abs(mx / N) < 5.0 * sqrt(1.0 / 3.0 / N) && abs(mz / N) < 5.0 * sqrt(1.0 / 3.0 / N) &&
          abs(mzz / N - 1.0 / 3.0) < 5.0 * sqrt(4.0 / 45.0 / N), "unit_vector3 moments");
    let zs = draw(N) {
        let x, y, z = random::unit_vector3(g);
        z
    };
    check_ks(zs) { x => (x + 1.0) / 2.0 };
    var v: f64[5] = [0.0, 0.0, 0.0, 0.0, 0.0];
    var s4 = 0.0;
    for i in 0..N {
        random::unit_vector(g, v);
        var n2 = 0.0;
        for x in v { n2 += x * x; }
        if abs(n2 - 1.0) > 1e-14 { ok = false; }
        s4 += v[4] * v[4];
    }
    check(ok, "unit_vector has length 1");
    check(abs(s4 / N - 0.2) < 0.01, "unit_vector coordinates");
    let rs = draw(N) {
        let x, y, z = random::in_ball3(g);
        sqrt(x * x + y * y + z * z)
    };
    check_ks(rs) { x => x * x * x };
    let ds = draw(N) {
        let x, y = random::in_disk(g);
        sqrt(x * x + y * y)
    };
    check_ks(ds) { x => x * x };

    group("multivariate normal");
    // Covariance [[4, 2], [2, 2]] = L L^T with L = [[2, 0], [1, 1]].
    let l = [2.0, 0.0, 1.0, 1.0];
    var o: f64[2] = [0.0, 0.0];
    var sxy = 0.0;
    var sxx = 0.0;
    var syy = 0.0;
    for i in 0..N {
        random::multivariate_normal(g, [1.0, -1.0], l, o);
        let x = o[0] - 1.0;
        let y = o[1] + 1.0;
        sxx += x * x;
        syy += y * y;
        sxy += x * y;
    }
    check(abs(sxx / N - 4.0) < 0.1 && abs(syy / N - 2.0) < 0.05 && abs(sxy / N - 2.0) < 0.06, "covariance");

    group("dirichlet");
    var dv: f64[3] = [0.0, 0.0, 0.0];
    var dm: f64[3] = [0.0, 0.0, 0.0];
    ok = true;
    for i in 0..N {
        random::dirichlet(g, [1.0, 2.0, 7.0], dv);
        let s = dv[0] + dv[1] + dv[2];
        if abs(s - 1.0) > 1e-14 { ok = false; }
        for x, j in dv { dm[j] += x; }
    }
    check(ok, "on the simplex");
    check(abs(dm[0] / N - 0.1) < 0.002 && abs(dm[2] / N - 0.7) < 0.002, "means");
    // Tiny concentrations: still on the simplex, nearly always one corner.
    ok = true;
    for i in 0..1000 {
        random::dirichlet(g, [1e-4, 1e-4, 1e-4], dv);
        let s = dv[0] + dv[1] + dv[2];
        if !(abs(s - 1.0) < 1e-14) { ok = false; }
    }
    check(ok, "tiny concentrations");
}

fn test_sampling() {
    var g = random::seed(7);
    group("shuffle");
    // Every one of the 24 orders of 4 equally often.
    var ks: i64[>..] = [];
    for i in 0..N {
        var p: i64[4] = [0, 1, 2, 3];
        random::shuffle(g, p);
        ks.push(p[0] * 64 + p[1] * 16 + p[2] * 4 + p[3]);
    }
    var codes: i64[>..] = [];
    for k in ks {
        // The rank of the permutation among the 24.
        let a = k >> 6;
        let b = (k >> 4) & 3;
        let c = (k >> 2) & 3;
        let rb = if b > a { b - 1 } else { b };
        var rc = c;
        if c > a { rc--; }
        if c > b { rc--; }
        codes.push(a * 6 + rb * 2 + rc);
    }
    check_pmf(codes, 0, 23, "uniform over orders") { k => 1.0 / 24.0 };
    let pm = random::permutation(g, 1000);
    var seen: bool[>..] = [];
    seen.resize(1000, false);
    for x in pm { seen[x] = true; }
    var ok = true;
    for s in seen { if !s { ok = false; } }
    check(ok, "permutation");

    group("sample_indices");
    for nk in [[10, 3], [10, 10], [1000000, 5], [100, 30]] {
        let n = nk[0];
        let k = nk[1];
        let s = random::sample_indices(g, n, k);
        var d = dictionary<i64, bool> {};
        ok = s.len == k;
        for x in s {
            if x < 0 || x >= n || !d.insert(x, true) { ok = false; }
        }
        check(ok, "distinct and in range");
    }
    // Each index equally likely to be chosen, and to come first.
    var firsts: i64[>..] = [];
    var all:    i64[>..] = [];
    for i in 0..N / 3 {
        let s = random::sample_indices(g, 20, 3);
        firsts.push(s[0]);
        for x in s { all.push(x); }
        let t = random::sample_indices(g, 100, 3);
        firsts.push(t[0] % 20);
    }
    check_pmf(firsts, 0, 19, "first uniform") { k => 1.0 / 20.0 };
    check_pmf(all, 0, 19, "membership uniform") { k => 1.0 / 20.0 };
}

fn main() {
    test_generators();
    test_uniform();
    test_integers();
    test_normal_exponential();
    test_continuous();
    test_discrete();
    test_multivariate();
    test_sampling();
    print(passed, " passed, ", failed, " failed");
    if failed > 0 { exit(1); }
}
