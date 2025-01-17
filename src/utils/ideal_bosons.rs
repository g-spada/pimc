use std::f64::consts::PI;

const ZETA_3_2: f64 = 2.612_375_348_685_488_f64;
//const ZETA_5_2: f64 = 1.341487257250917179756769693348_f64;

#[cfg(test)]
/// Computes the single particle functions z(k \beta) and their derivatives dz(k\beta)/d\beta for a
/// system of `npart` bosons in a cubic box of side L with periodic boundary conditions, and at the temperature
/// T = 1/(k_B \beta) measured in units of the ideal gas BEC condensation temperature Tc0.
/// Tc0 = 2 \pi \hbar^2 / m * (( npart/L^3 ) / \zeta(3/2))^(3/2)
///
/// The single particle functions z(k \beta) are defined as:
/// z(k \beta) = \sum_{nx, ny, nz} exp( - k \beta \varepsilon(nx, ny, nz) )
///
/// where the sum is carried over nx, ny and nz is over 0, \pm 1, \pm 2, ...
/// and the single particle energies \varepsilon(nx, ny, nz) given by
/// \varepsilon(nx, ny, nz) = (2 \pi)^2 \hbar^2 / (2 m L^2) * (nx^2 + ny^2 + nz^2)
///
/// # Arguments
/// * `npart` - Total number of particles.
/// * `temp` - Temperature in units of Tc0.
///
/// # Returns
/// * A tuple of vectors containing the z(k \beta) and dz(k\beta)/d\beta. The index labels k and
///   goes from 0 to npart included. The k=0 terms are canonically set to 1 and 0 respectively.
///   - Values of dz/d\beta are given in units of kB*Tc0
///
/// # Implementation details
/// * Raw (unoptimized) version that sums all the values without exploiting symmetries.
/// * For test purposes.
fn single_particle_z_dz_raw(
    npart: usize, // Total number of particles
    temp: f64,    // Temperature in units of Tc0
) -> (Vec<f64>, Vec<f64>) {
    assert!(temp > 0.0, "Temperature must be non-zero and positve.");
    const NMAX: i32 = 20;
    // Determine the prefactor  (2 \pi)^2 \hbar^2 / (2 m L^2) in units of kB Tc0
    let energy_constant = PI * (ZETA_3_2 / npart as f64).powf(2.0 / 3.0);
    let mut single_particle_z = vec![0.0; npart + 1];
    let mut single_particle_dz = vec![0.0; npart + 1];
    // FULL UNOPTIMIZED COMPUTATION (SLOW)
    for n1 in -NMAX..=NMAX {
        for n2 in -NMAX..=NMAX {
            for n3 in -NMAX..=NMAX {
                let n1sq = (n1 as f64).powf(2.0);
                let n2sq = (n2 as f64).powf(2.0);
                let n3sq = (n3 as f64).powf(2.0);
                let n_squared = n1sq + n2sq + n3sq;
                let epsilon_n = energy_constant * n_squared;
                let exp_neg_beta_epsilon_n = (-epsilon_n / temp).exp();
                for k in 0..=npart {
                    let k_f = k as f64;
                    let exp_term = exp_neg_beta_epsilon_n.powf(k_f);
                    single_particle_z[k] += exp_term;
                    single_particle_dz[k] += exp_term * (-epsilon_n * k_f);
                }
            }
        }
    }
    single_particle_z[0] = 1.0;
    single_particle_dz[0] = 0.0;
    (single_particle_z, single_particle_dz)
}

/// Computes the single particle functions z(k \beta) and their derivatives dz(k\beta)/d\beta for a
/// system of `npart` bosons in a cubic box of side L with periodic boundary conditions, and at the temperature
/// T = 1/(k_B \beta) measured in units of the ideal gas BEC condensation temperature Tc0.
/// Tc0 = 2 \pi \hbar^2 / m * (( npart/L^3 ) / \zeta(3/2))^(3/2)
///
/// The single particle functions z(k \beta) are defined as:
/// z(k \beta) = \sum_{nx, ny, nz} exp( - k \beta \varepsilon(nx, ny, nz) )
///
/// where the sum is carried over nx, ny and nz is over 0, \pm 1, \pm 2, ...
/// and the single particle energies \varepsilon(nx, ny, nz) given by
/// \varepsilon(nx, ny, nz) = (2 \pi)^2 \hbar^2 / (2 m L^2) * (nx^2 + ny^2 + nz^2)
///
/// # Arguments
/// * `npart` - Total number of particles.
/// * `temp` - Temperature in units of Tc0.
///
/// # Returns
/// * A tuple of vectors containing the z(k \beta) and dz(k\beta)/d\beta. The index labels k and
///   goes from 0 to npart included. The k=0 terms are canonically set to 1 and 0 respectively.
///   - Values of dz/d\beta are given in units of kB*Tc0
pub fn single_particle_z_dz(
    npart: usize, // Total number of particles
    temp: f64,    // Temperature in units of Tc0
) -> (Vec<f64>, Vec<f64>) {
    assert!(temp > 0.0, "Temperature must be non-zero and positve.");
    let energy_constant = PI * (ZETA_3_2 / npart as f64).powf(2. / 3.);
    // Dynamically determine the value of nmax based on the input parameters
    const MAX_EXPONENT: f64 = 400.0;
    const NMAX_LOWER_BOUND: usize = 10;
    let nmax: usize =
        ((MAX_EXPONENT / (energy_constant / temp)).sqrt().ceil() as usize).min(NMAX_LOWER_BOUND);
    let mut single_particle_z = vec![1.0; npart + 1]; // case (nx=0,ny=0,nz=0) already included
    let mut single_particle_dz = vec![0.0; npart + 1];
    // We sum only on positive numbers and account for the proper symmetry factor
    for n1 in 1..=nmax {
        // n1 != 0, n2 = 0, n3 = 0
        let n1sq = (n1 as f64).powf(2.0);
        let epsilon_n = energy_constant * n1sq;
        let exp_neg_beta_epsilon_n = (-epsilon_n / temp).exp();
        for (k, (z, dz)) in single_particle_z
            .iter_mut()
            .zip(single_particle_dz.iter_mut())
            .enumerate()
            .skip(1)
        {
            let k_f = k as f64;
            let exp_term = exp_neg_beta_epsilon_n.powf(k_f);
            // The factor of 6 = 3 * 2 is obtained the 3 possible choices of selecting
            // n1 out of (nx, ny, nz) and the 2 possibilities of positive and negative integers.
            *z += 6.0 * exp_term;
            *dz += 6.0 * exp_term * (-k_f * epsilon_n);
        }
        // n1 != 0, n2 != 0, n3 = 0
        for n2 in n1..=nmax {
            // n2 is further restricted to be n2 >= n1
            // The reordering symmetry is taken into account the following factor
            let factor = if n1 == n2 { 1.0 } else { 2.0 };
            let n1sq = (n1 as f64).powf(2.0);
            let n2sq = (n2 as f64).powf(2.0);
            let epsilon_n = energy_constant * (n1sq + n2sq);
            let exp_neg_beta_epsilon_n = (-epsilon_n / temp).exp();
            for (k, (z, dz)) in single_particle_z
                .iter_mut()
                .zip(single_particle_dz.iter_mut())
                .enumerate()
                .skip(1)
            {
                let k_f = k as f64;
                let exp_term = exp_neg_beta_epsilon_n.powf(k_f);
                // The factor of 12 = 3 * 2 * 2 is obtained the 3 possible choices of selecting
                // (n1, n2) out of (nx, ny, nz) and the 2 possibilities of positive and negative
                // integers for each of n1 and n2.
                *z += factor * 12.0 * exp_term;
                *dz += factor * 12.0 * exp_term * (-k_f * epsilon_n);
            }
            // n1 != 0, n2 != 0, n3 != 0
            for n3 in n2..=nmax {
                // n3 is further restricted to be n3 >= n2
                // The reordering symmetry is taken into account by the following factor
                let factor;
                if n1 == n2 && n2 == n3 {
                    factor = 1.0;
                } else if n1 < n2 && n2 < n3 {
                    factor = 6.0;
                } else if n1 == n2 || n2 == n3 {
                    factor = 3.0;
                } else {
                    factor = 1.0
                };
                let n3sq = (n3 as f64).powf(2.0);
                let epsilon_n = energy_constant * (n1sq + n2sq + n3sq);
                let exp_neg_beta_epsilon_n = (-epsilon_n / temp).exp();
                for (k, (z, dz)) in single_particle_z
                    .iter_mut()
                    .zip(single_particle_dz.iter_mut())
                    .enumerate()
                    .skip(1)
                {
                    let k_f = k as f64;
                    let exp_term = exp_neg_beta_epsilon_n.powf(k_f);
                    // The factor of 8 = 2 * 2 * 2 is obtained the 2 possibilities of positive and negative
                    // integers for each of n1, n2 and n3.
                    *z += factor * 8.0 * exp_term;
                    *dz += factor * 8.0 * exp_term * (-k_f * epsilon_n);
                }
            }
        }
    }
    single_particle_z[0] = 1.0;
    single_particle_dz[0] = 0.0;
    (single_particle_z, single_particle_dz)
}

/// Computes the logarithm of the canonical partition functions ln(Z_N) and their
/// derivatives d(ln(Z_N))/d\beta.
///
/// # Arguments
/// * `single_particle_z` - a vector containing the single particle functions z(k \beta)
/// * `deriv_single_particle_z` - a vector containing the derivative of the single particle
///   functions with respect to \beta
///
/// # Returns
/// - A tuple of vectors, the first containing the logarithm of the N-particles partition functions
///   ln(Z_N(\beta)), the second containing their derivatives with respect to \beta
///
/// # Implementation Details
/// * Makes use of the recursion relations Z_N = \sum_{k=1}^N z(k\beta) Z_{N-k}.
/// * Works with logarithms to ensure numerical stability.
///
/// # References
/// *  P. Borrmann and F. Gert, "Recursion formulas for quantum statistical partition functions",
///    The Journal of Chemical Physics 98, 2484 (1993)
/// * K. Werner, "Statistical Mechanics Algorithms and Computations", Oxford University Press (2006).
pub fn compute_ln_partition_functions(
    single_particle_z: Vec<f64>,
    deriv_single_particle_z: Vec<f64>,
) -> (Vec<f64>, Vec<f64>) {
    assert_eq!(
        single_particle_z.len(),
        deriv_single_particle_z.len(),
        "Lengths should match"
    );
    let mut ln_partition_function = vec![0.0_f64; single_particle_z.len()];
    let mut deriv_ln_partition_function = vec![0.0_f64; single_particle_z.len()];
    // Recursion relations start with Z_0 = 1 (thus ln Z_0 = 0 and dlnZ_0/d\beta = 0)
    // Settig values for N = 1
    let z1val = single_particle_z[1];
    let dz1val = deriv_single_particle_z[1];
    ln_partition_function[1] = z1val.ln();
    deriv_ln_partition_function[1] = dz1val / z1val;
    // Use recursion relations to compute values for N > 1
    for i in 2..single_particle_z.len() {
        let mut zz_sum: f64 = 0.0;
        let mut dzz_sum: f64 = 0.0;
        for k in 2..=i {
            // Values of single-particle z(k \beta) and dz(k\beta)/d\beta
            let zkval = single_particle_z[k];
            let dzkval = deriv_single_particle_z[k];
            let zz_k = (zkval / z1val)
                * (ln_partition_function[i - k] - ln_partition_function[i - 1]).exp();
            zz_sum += zz_k;
            dzz_sum += zz_k
                * (dzkval / zkval - dz1val / z1val + deriv_ln_partition_function[i - k]
                    - deriv_ln_partition_function[i - 1]);
        }
        ln_partition_function[i] =
            -(i as f64).ln() + z1val.ln() + ln_partition_function[i - 1] + (1.0 + zz_sum).ln();
        deriv_ln_partition_function[i] =
            dz1val / z1val + deriv_ln_partition_function[i - 1] + 1.0 / (1.0 + zz_sum) * dzz_sum;
    }
    (ln_partition_function, deriv_ln_partition_function)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_single_particle_partition_function() {
        const N: usize = 10; // Number of bosons

        let temps = [0.1, 0.5, 1.0, 1.2, 1.5];
        for temp in temps {
            let (z1, dz1) = single_particle_z_dz(N, temp);
            let (z1_raw, dz1_raw) = single_particle_z_dz_raw(N, temp);
            //println!("temp = {temp}");
            //println!("{z1:#?}");
            //println!("{z1_raw:#?}");
            //let diff_z1: Vec<f64> = z1
            //.iter().skip(1)
            //.zip(z1_raw.iter().skip(1))
            //.map(|(a, b)| (1.0 - b / a))
            //.collect();
            //println!("{diff_z1:#?}");
            //println!("{dz1:#?}");
            //println!("{dz1_raw:#?}");
            assert_eq!(z1.len(), z1_raw.len());
            //let diff_dz1: Vec<f64> = dz1
            //.iter().skip(1)
            //.zip(dz1_raw.iter().skip(1))
            //.map(|(a, b)| (1.0 - b / a))
            //.collect();
            //println!("{diff_dz1:#?}");
            assert!(z1
                .iter()
                .skip(1)
                .zip(z1_raw.iter().skip(1))
                .all(|(a, b)| (1.0 - b / a).abs() < 1e-12));
            assert!(dz1
                .iter()
                .skip(1)
                .zip(dz1_raw.iter().skip(1))
                .all(|(a, b)| (1.0 - b / a).abs() < 1e-12));
        }
    }

    #[test]
    fn test_partition_function() {
        const N: usize = 10; // Number of bosons
        let temps = [0.1, 0.5, 1.0, 1.2, 1.5];
        const E_EXPECTED: [f64; 5] = [
            2.0467686071747325e-6,
            0.08378741697476733,
            0.6642656699606142,
            1.0434331061799722,
            1.6418216784919957,
        ];
        for (&temp, &expected) in temps.iter().zip(E_EXPECTED.iter()) {
            let (z1, dz1) = single_particle_z_dz(N, temp);
            let (_lnz, dlnz) = compute_ln_partition_functions(z1, dz1);
            let energy = -dlnz[N] / N as f64;
            //println!("{:#?}", lnz);
            //println!("{:#?}", dlnz);
            //println!("{}", energy);
            //println!("{}", expected);
            assert!((energy - expected).abs() < 1e-12);
        }
    }
}
