use ndarray::{ArrayBase, DataMut, Ix2};
use rand::Rng;
use rand_distr::{Distribution, Normal};

/// Redraws the path between the initial and final slices in `delta_j` links
/// using Levy's staging algorithm. This function samples new positions for the
/// beads at intermediate slices, while keeping the initial and final beads fixed.
///
/// The algorithm samples the free density matrix using Gaussian distributions
/// to efficiently handle the kinetic term of the quantum propagator.
///
/// # Parameters
/// - `polymer`: A mutable reference to a 2D array (`ndarray::ArrayBase`)
///   representing the polymer path. Each row corresponds to a time slice, and
///   each column corresponds to a spatial dimension. The initial and final rows
///   (slices) remain fixed, while intermediate slices are redrawn.
/// - `two_lambda_tau`: A positive float representing the product of
///   \( 2 \lambda = \hbar^2 / ( 2 * m ) \) (related to the quantum kinetic energy scale)
///   and the imaginary-time step \( \tau \).
/// - `rng`: A mutable reference to a random number generator implementing the
///   `rand::Rng` trait, used to generate random numbers for Gaussian sampling.
///
/// # Panics
/// - If `two_lambda_tau` is non-positive.
/// - If the shape of `polymer` is not 2D.
/// - If the number of slices in `polymer` is less than 2.
///
/// # References
/// - W. Krauth, "Statistical Mechanics: Algorithms and Computations", OUP Oxford, 2006, [<https://doi.org/10.1093/oso/9780198515357.001.0001>], Algorithm 3.5, p.154 (with different normalization).
/// - Condens. Matter 2022, 7, 30, Eq. (27) [<http://arxiv.org/abs/2203.00010>]
///   *Note*: This reference contains a typo in the last denominator (extra π).
///
/// # Example
/// ```rust
/// use pimc_rs::updates::redraw_staging::redraw_staging;
/// use ndarray::Array2;
/// use rand::thread_rng;
///
/// // Initialize a polymer with 3 time slices and 2 spatial dimensions
/// let mut polymer = Array2::from_shape_vec((3, 2), vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0]).unwrap();
/// let two_lambda_tau = 1.0;
/// let mut rng = thread_rng();
///
/// // Modify the intermediate slice using Levy's staging algorithm
/// redraw_staging(&mut polymer, two_lambda_tau, &mut rng);
///
/// // Check that the first and last slices are unchanged
/// assert_eq!(polymer.row(0).to_vec(), vec![0.0, 0.0]);
/// assert_eq!(polymer.row(2).to_vec(), vec![2.0, 2.0]);
///
/// // Intermediate slice should be modified
/// assert_ne!(polymer.row(1).to_vec(), vec![1.0, 1.0]);
/// ```
pub fn redraw_staging<S, R: Rng>(polymer: &mut ArrayBase<S, Ix2>, two_lambda_tau: f64, rng: &mut R)
where
    S: DataMut<Elem = f64>,
{
    debug_assert!(two_lambda_tau >= 0.0, "two_lambda_tau cannot be negative");

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

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;
    use rand::thread_rng;

    #[test]
    fn test_redraw_staging_basic() {
        let mut polymer =
            Array2::from_shape_vec((3, 2), vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0]).unwrap();
        let two_lambda_tau = 1.0;
        let mut rng = thread_rng();

        // Modify the intermediate slice
        redraw_staging(&mut polymer, two_lambda_tau, &mut rng);

        // First and last slices should remain unchanged
        assert_eq!(polymer.row(0).to_vec(), vec![0.0, 0.0]);
        assert_eq!(polymer.row(2).to_vec(), vec![2.0, 2.0]);

        // Intermediate slice should change
        assert_ne!(polymer.row(1).to_vec(), vec![1.0, 1.0]);
    }

    #[test]
    fn test_redraw_staging_deterministic() {
        let mut polymer = Array2::from_shape_vec(
            (5, 2),
            vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, -8.0, 8.0],
        )
        .unwrap();

        // With vanishing two_lambda_tau the sampled positions are deterministic
        let two_lambda_tau = 0.0;
        let mut rng = thread_rng();

        // Modify the intermediate slice
        redraw_staging(&mut polymer, two_lambda_tau, &mut rng);

        // First and last slices should remain unchanged
        assert_eq!(polymer.row(0).to_vec(), vec![0.0, 0.0]);
        assert_eq!(polymer.row(4).to_vec(), vec![-8.0, 8.0]);

        // Intermediate slices must lay on a straight line
        assert_eq!(polymer.row(1).to_vec(), vec![-2.0, 2.0]);
        assert_eq!(polymer.row(2).to_vec(), vec![-4.0, 4.0]);
        assert_eq!(polymer.row(3).to_vec(), vec![-6.0, 6.0]);
    }

    #[test]
    #[should_panic]
    fn test_redraw_staging_invalid_two_lambda_tau() {
        let mut polymer = Array2::zeros((3, 2));
        let two_lambda_tau = -1.0; // Invalid value
        let mut rng = thread_rng();

        // This should panic
        redraw_staging(&mut polymer, two_lambda_tau, &mut rng);
    }

    #[test]
    #[should_panic(expected = "The polymer must have at lesat 2 slices")]
    fn test_redraw_staging_insufficient_slices() {
        let mut polymer = Array2::zeros((1, 2)); // Only one slice
        let two_lambda_tau = 1.0;
        let mut rng = thread_rng();

        // This should panic
        redraw_staging(&mut polymer, two_lambda_tau, &mut rng);
    }

    #[test]
    fn test_redraw_staging_multi_dimension() {
        let mut polymer = Array2::from_shape_vec(
            (4, 3),
            vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0],
        )
        .unwrap();
        let two_lambda_tau = 2.0;
        let mut rng = thread_rng();

        // Modify intermediate slices
        redraw_staging(&mut polymer, two_lambda_tau, &mut rng);

        // Ensure boundary slices are unchanged
        assert_eq!(polymer.row(0).to_vec(), vec![0.0, 0.0, 0.0]);
        assert_eq!(polymer.row(3).to_vec(), vec![3.0, 3.0, 3.0]);

        // Ensure intermediate slices are modified
        assert_ne!(polymer.row(1).to_vec(), vec![1.0, 1.0, 1.0]);
        assert_ne!(polymer.row(2).to_vec(), vec![2.0, 2.0, 2.0]);
    }

    #[test]
    fn test_redraw_staging_first_column() {
        use ndarray::s;

        let mut polymer = Array2::from_shape_vec(
            (4, 3),
            vec![
                0.0, 0.0, 0.0, // Slice 0
                1.0, 1.0, 1.0, // Slice 1
                2.0, 2.0, 2.0, // Slice 2
                3.0, 3.0, 3.0, // Slice 3
            ],
        )
        .unwrap();
        let two_lambda_tau = 2.0;
        let mut rng = thread_rng();

        // Extract only the first column as a mutable view
        {
            let mut first_column = polymer.slice_mut(s![.., 0..1]);
            redraw_staging(&mut first_column, two_lambda_tau, &mut rng);
        }

        // Ensure boundary slices (first and last rows) are unchanged
        assert_eq!(polymer[[0, 0]], 0.0); // First slice, first column
        assert_eq!(polymer[[3, 0]], 3.0); // Last slice, first column

        // Ensure intermediate slices in the first column are modified
        assert_ne!(polymer[[1, 0]], 1.0); // Second slice, first column (modified)
        assert_ne!(polymer[[2, 0]], 2.0); // Third slice, first column (modified)

        // Ensure other columns are unaffected
        assert_eq!(polymer.column(1).to_vec(), vec![0.0, 1.0, 2.0, 3.0]); // Second column
        assert_eq!(polymer.column(2).to_vec(), vec![0.0, 1.0, 2.0, 3.0]); // Third column
    }
}
