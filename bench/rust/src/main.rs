// rand 0.9 and rand_distr 0.5 on the cases of bench.goose: ns per value,
// filling a Vec of 2^20 values, the best of at least 5 samples and 0.2 s.
// The generator is rand_xoshiro's Xoshiro256PlusPlus, the same as Goose's
// default, so the comparison is of the distributions.
//
//   cargo run --release -- [out.csv]        (from bench/rust)
//   RUSTFLAGS="-C target-cpu=native" for the AVX2 build

use rand::seq::SliceRandom;
use rand::{Rng, RngCore, SeedableRng};
use rand_distr::{
    Beta, Binomial, Cauchy, ChiSquared, Distribution, Exp1, Gamma, Geometric, Hypergeometric,
    LogNormal, Poisson, StandardNormal, StudentT, Weibull, weighted::WeightedAliasIndex,
};
use rand_xoshiro::Xoshiro256PlusPlus;
use std::hint::black_box;
use std::time::Instant;

const N: usize = 1 << 20;

fn time_it(lines: &mut Vec<String>, name: &str, mut f: impl FnMut()) {
    let (mut best, mut total, mut samples) = (f64::MAX, 0.0, 0);
    while samples < 5 || total < 0.2 {
        let t0 = Instant::now();
        f();
        let dt = t0.elapsed().as_secs_f64();
        best = best.min(dt);
        total += dt;
        samples += 1;
    }
    let ns = best / N as f64 * 1e9;
    println!("{name}: {ns:.3} ns");
    lines.push(format!("{name},{ns}"));
}

fn floats<D: Distribution<f64>>(lines: &mut Vec<String>, g: &mut Xoshiro256PlusPlus, out: &mut [f64], name: &str, d: D) {
    time_it(lines, name, || {
        for x in out.iter_mut() {
            *x = d.sample(g);
        }
        black_box(&out);
    });
}

fn ints<D: Distribution<u64>>(lines: &mut Vec<String>, g: &mut Xoshiro256PlusPlus, out: &mut [u64], name: &str, d: D) {
    time_it(lines, name, || {
        for x in out.iter_mut() {
            *x = d.sample(g);
        }
        black_box(&out);
    });
}

fn main() {
    let path = std::env::args().nth(1).unwrap_or("bench_rust.csv".into());
    let mut lines = Vec::new();
    let mut g = Xoshiro256PlusPlus::seed_from_u64(1);
    let mut f = vec![0.0f64; N];
    let mut u = vec![0u64; N];
    let mut idx: Vec<u64> = (0..N as u64).collect();
    let l = &mut lines;

    time_it(l, "raw xoshiro256++", || {
        for x in u.iter_mut() {
            *x = g.next_u64();
        }
        black_box(&u);
    });
    time_it(l, "uniform", || {
        for x in f.iter_mut() {
            *x = g.random::<f64>();
        }
        black_box(&f);
    });
    floats(l, &mut g, &mut f, "normal", StandardNormal);
    floats(l, &mut g, &mut f, "exponential", Exp1);
    time_it(l, "below 1000", || {
        for x in u.iter_mut() {
            *x = g.random_range(0..1000u64);
        }
        black_box(&u);
    });
    time_it(l, "below 3e18", || {
        for x in u.iter_mut() {
            *x = g.random_range(0..3000000000000000000u64);
        }
        black_box(&u);
    });
    floats(l, &mut g, &mut f, "gamma 0.5", Gamma::new(0.5, 1.0).unwrap());
    floats(l, &mut g, &mut f, "gamma 3", Gamma::new(3.0, 1.0).unwrap());
    floats(l, &mut g, &mut f, "beta 2 5", Beta::new(2.0, 5.0).unwrap());
    floats(l, &mut g, &mut f, "beta 0.5 0.5", Beta::new(0.5, 0.5).unwrap());
    floats(l, &mut g, &mut f, "chi_squared 5", ChiSquared::new(5.0).unwrap());
    floats(l, &mut g, &mut f, "student_t 5", StudentT::new(5.0).unwrap());
    floats(l, &mut g, &mut f, "lognormal", LogNormal::new(0.0, 1.0).unwrap());
    floats(l, &mut g, &mut f, "cauchy", Cauchy::new(0.0, 1.0).unwrap());
    floats(l, &mut g, &mut f, "weibull 1.5", Weibull::new(1.0, 1.5).unwrap());
    floats(l, &mut g, &mut f, "poisson 3", Poisson::new(3.0).unwrap());
    floats(l, &mut g, &mut f, "poisson 1000", Poisson::new(1000.0).unwrap());
    ints(l, &mut g, &mut u, "binomial 20 0.3", Binomial::new(20, 0.3).unwrap());
    ints(l, &mut g, &mut u, "binomial 1000 0.3", Binomial::new(1000, 0.3).unwrap());
    ints(l, &mut g, &mut u, "geometric 0.01", Geometric::new(0.01).unwrap());
    ints(l, &mut g, &mut u, "hypergeometric 500 500 200", Hypergeometric::new(1000, 500, 200).unwrap());
    let w = WeightedAliasIndex::new(vec![0.1, 0.2, 0.3, 0.15, 0.05, 0.2]).unwrap();
    time_it(l, "alias 6", || {
        for x in u.iter_mut() {
            *x = w.sample(&mut g) as u64;
        }
        black_box(&u);
    });
    time_it(l, "shuffle", || {
        idx.shuffle(&mut g);
        black_box(&idx);
    });
    std::fs::write(&path, lines.join("\n") + "\n").unwrap();
}
