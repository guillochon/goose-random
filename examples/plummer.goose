// Initial conditions for an N-body run: a Plummer sphere in Henon units
// (G = M = 1, total energy -1/4), positions by inverting the cumulative mass
// profile, speeds by von Neumann rejection from the distribution function,
// and isotropic directions for both. Prints the virial ratio, which should
// be near 1/2, and the half-mass radius, near 0.769 in these units.
//
//   goose examples/plummer.goose [-- n seed]

import std;
import os;
import math;
import random from "..";

struct body { x: f64, y: f64, z: f64, vx: f64, vy: f64, vz: f64 }

fn plummer<G>(g: G&, n: i64) -> body[>..] {
    let a = 3.0 * PI / 16.0;            // the scale radius for E = -1/4
    var out: body[>..] = [];
    for i in 0..n {
        // Radius: M(r) = r^3 / (r^2 + a^2)^(3/2), inverted.
        let m = random::uniform_pos(g);
        let r = a / sqrt(pow(m, -2.0 / 3.0) - 1.0);
        let ux, uy, uz = random::unit_vector3(g);
        // Speed: q = v / v_escape has density q^2 (1 - q^2)^(7/2).
        var q = 0.0;
        loop {
            q = random::uniform(g);
            if 0.1 * random::uniform(g) < q * q * pow(1.0 - q * q, 3.5) { break; }
        }
        let v = q * sqrt(2.0) * pow(r * r + a * a, -0.25);
        let wx, wy, wz = random::unit_vector3(g);
        out.push(body { r * ux, r * uy, r * uz, v * wx, v * wy, v * wz });
    }
    return out;
}

fn main() {
    let args = args();
    let n = if args.len > 1 { parse_int(args[1]) } else { 4000 };
    let seed = if args.len > 2 { parse_int(args[2]) } else { 1 };
    var g = random::seed(seed as u64);
    let bodies = plummer(g, n);

    let m = 1.0 / n;
    var kin = 0.0;
    var pot = 0.0;
    var radii: f64[>..] = [];
    for b, i in bodies {
        kin += 0.5 * m * (b.vx * b.vx + b.vy * b.vy + b.vz * b.vz);
        radii.push(sqrt(b.x * b.x + b.y * b.y + b.z * b.z));
        for j in i + 1..n {
            let c = bodies[j];
            let dx = b.x - c.x;
            let dy = b.y - c.y;
            let dz = b.z - c.z;
            pot -= m * m / sqrt(dx * dx + dy * dy + dz * dz);
        }
    }
    sort(radii);
    print(n, " bodies: kinetic ", kin, ", potential ", pot, ", virial ratio ", -kin / pot,
          ", half-mass radius ", radii[n / 2]);
}
