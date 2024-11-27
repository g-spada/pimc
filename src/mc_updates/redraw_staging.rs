use ndarray::{ArrayBase, DataMut, Ix2};
use rand::Rng;
use rand_distr::{Distribution, Normal};

// ========================================================================
// Levy staging function
// ========================================================================
// Redraws the path between the initial and final slices in delta_j links.
// sampling the free density matrix gaussian distributions.
// Initial and final beads and are kept fixed.
// Exactly delta_j time slices are modified.
// Information about the algorithm can be found
// @ alg3.5 (p.154) of "Algorithm and Computations",
//   W. Krauth (with different normalization);
// @ Eq.(27) of Condens. Matter 2022, 7, 30
//   [http://arxiv.org/abs/2203.00010]
//   notice a typo: extra pi at the last denominator
pub fn redraw_staging<S, R: Rng>(polymer: &mut ArrayBase<S, Ix2>, two_lambda_tau: f64, rng: &mut R)
where
    S: DataMut<Elem = f64>,
{
    debug_assert!(two_lambda_tau > 0.0, "tau_mass_ratio must be positive");

    let shape = polymer.shape();
    debug_assert_eq!(shape.len(), 2, "Expected a tensor with 2 indices");
    let n_slices = shape[0];
    let n_dimens = shape[1];
    debug_assert!(n_slices > 1, "The polymer must have at lesat 2 slices");
    let delta_j = n_slices - 1; // The number of links

    // Normal distribution (mean = 0, std_dev = 1)
    let normal = Normal::new(0.0, 1.0).unwrap();

    // Sampling of the (delta_j - 1) intermediate time-slices
    for j in 1..delta_j {
        let j_dist = (delta_j - j) as f64;
        let j_dist_plus = j_dist + 1.0;
        let aj = j_dist / j_dist_plus;
        let sigma = (two_lambda_tau * aj).sqrt();
        for d in 0..n_dimens {
            let r_j_star: f64 =
                (polymer[[n_slices - 1, d]] + j_dist * polymer[[j - 1, d]]) / j_dist_plus;
            let delta_r: f64 = sigma * normal.sample(rng);
            polymer[[j, d]] = r_j_star + delta_r;
        }
    }
}
