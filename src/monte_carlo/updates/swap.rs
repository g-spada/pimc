use crate::action::traits::PotentialDensityMatrix;
use crate::impl_tunable_parameters;
use crate::monte_carlo::accepted_update::AcceptedUpdate;
use crate::monte_carlo::levy_staging::levy_staging;
use crate::monte_carlo::proposed_update::ProposedUpdate;
use crate::monte_carlo::traits::{MonteCarloStep, MonteCarloTunable};
use crate::path::traits::{
    WorldLineBatchPositions, WorldLineDimensions, WorldLinePermutationAccess,
    WorldLinePositionAccess, WorldLineStateEq, WorldLineWormAccess,
};
use crate::space::traits::Space;
use crate::system::traits::SystemAccess;
use log::{debug, trace};
use ndarray::{Array1, Array2};
use rand_distr::weighted::WeightedIndex;
use rand_distr::Distribution;

/// A Monte Carlo update that glues a polymer to the tail of the worm, thus sampling permutations.
///
/// # Fields
/// - `min_delta_t`: Minimum time slices to redraw. Must be greater than 1.
/// - `max_delta_t`: Maximum time slices to redraw. Must be greater than or equal to `min_delta_t`.
///
/// # Implementation Details
/// - It takes into account the space periodicity to satisfy the detailed balance condition.
/// - It doesn't require the polymers to have the initial slice within the fundamental cell.
#[derive(Debug, Clone, Copy)]
pub struct Swap {
    /// The minimum extent of the segment to redraw, in time slices.
    /// Must be greater than 1.
    min_delta_t: usize,

    /// The maximum extent of the segment to redraw, in time slices.
    /// Must be greater than or equal to `min_delta_t`.
    max_delta_t: usize,
}

impl Swap {
    pub fn new(min_delta_t: usize, max_delta_t: usize) -> Self {
        Self {
            min_delta_t,
            max_delta_t,
        }
    }
}

impl<S, A, R> MonteCarloStep<S, A, R> for Swap
where
    S: SystemAccess,
    S::WorldLine: WorldLineDimensions
        + WorldLinePositionAccess
        + WorldLineWormAccess
        + WorldLinePermutationAccess
        + WorldLineBatchPositions
        + WorldLineStateEq,
    A: PotentialDensityMatrix,
    R: rand::Rng,
{
    fn step(&mut self, system: &mut S, action: &A, rng: &mut R) -> Option<AcceptedUpdate> {
        debug!("Trying update");

        let tot_slices = S::WorldLine::TIME_SLICES;
        let tot_directions = S::WorldLine::SPATIAL_DIMENSIONS;
        let worldlines = system.path();
        let tot_particles = worldlines.particles();
        // Find head of the worm
        let tail = if let Some(tail) = worldlines.worm_tail() {
            tail
        } else {
            debug!("No worm tail present");
            return None;
        };
        let head = worldlines.worm_head().unwrap();

        // Randomly select the extent of the polymer to redraw
        let delta_t: usize = rng.random_range(self.min_delta_t..=self.max_delta_t);

        // Pivot slice
        let t0 = tot_slices - delta_t - 1;

        // Get two_lambda_tau value for particle
        let two_lambda_tau = system.two_lambda_tau(tail);

        // Compute the differences from each polymer at slice `t0` to the image of the tail that is
        // closer to the periodicity slice `tot_slices - 1`.
        // Find closest image of the tail
        let closest_tail_image = &worldlines.positions_at_time_slice(tot_slices - 1)
            - system.space().differences_from_reference(
                worldlines.positions_at_time_slice(tot_slices - 1),
                worldlines.position(tail, 0),
            );

        // Compute the weights
        let mut weights: Array1<f64> = worldlines
            .positions_at_time_slice(t0)
            .outer_iter()
            .zip(closest_tail_image.outer_iter())
            .map(|(r_pivot, r_tail)| {
                debug_assert_eq!(
                    r_tail.len(),
                    tot_directions,
                    "weights: row size must be {}, found {}",
                    tot_directions,
                    r_pivot.len()
                );
                let diff = &r_pivot - &r_tail;
                let distance_sq = diff.iter().map(|x| x * x).sum::<f64>();
                //let distance_sq = norm_l2(&row); // Compute Euclidean modulus (L2 norm)
                (-distance_sq / (2.0 * two_lambda_tau * delta_t as f64)).exp()
            })
            .collect();

        // Cannot swap with head (would lose the worm)
        weights[head] = 0.0;

        // Swap only allowed on beads with the same internal state
        for (i, weight) in weights.iter_mut().enumerate() {
            if !worldlines.beads_state_eq(tail, 0, i, tot_slices - 1) {
                *weight = 0.0;
            }
        }

        trace!("Weights: {:?}", weights);
        let sum_weights: f64 = weights.sum();
        if !sum_weights.is_normal() {
            debug!("Invalid weights. Not possible to swap.");
            return None;
        }

        // Select the particle by tower sampling from the discrete probability distribution given by
        // weights
        let p0 = WeightedIndex::new(&weights).unwrap().sample(rng);
        debug_assert!(p0 != head, "Cannot swap with head");
        trace!("Selected P{}, from pivot slice T{}", p0, t0,);

        // Compute weight from inverse move
        let tail_new = worldlines.following(p0).unwrap(); // Cannot fail: p0 is not head
        let mut sum_weights_inverse: f64 = 0.0;
        for particle in 0..tot_particles {
            if particle != head {
                let r_periodicity = worldlines.position(particle, tot_slices - 1);
                let r_tail_new = worldlines.position(tail_new, 0);
                let r_tail_image =
                    &r_periodicity - &system.space().difference(r_periodicity, r_tail_new);
                let r_pivot = worldlines.position(particle, t0);
                let distance_sq = (&r_pivot - &r_tail_image)
                    .iter()
                    .map(|x| x * x)
                    .sum::<f64>();
                sum_weights_inverse +=
                    (-distance_sq / (2.0 * two_lambda_tau * delta_t as f64)).exp();
            }
        }
        if !sum_weights_inverse.is_normal() {
            debug!("Invalid inverse weights. Not possible to swap.");
            return None;
        }

        let mut redraw_segment = Array2::<f64>::zeros((delta_t + 1, tot_directions));
        // Copy the initial bead (p0,t0)
        redraw_segment
            .row_mut(0)
            .assign(&worldlines.position(p0, t0));

        // Final bead: glue to previous tail
        redraw_segment
            .row_mut(delta_t)
            .assign(&closest_tail_image.row(p0));

        debug_assert!(system
            .space()
            .difference(redraw_segment.row(delta_t), worldlines.position(tail, 0))
            .iter()
            .all(|delta_r| delta_r.abs() < 1e-10));

        // We now redraw the segment t0..tot_slices with levy
        if delta_t > 1 {
            // Apply staging on the rest of the segment
            levy_staging(&mut redraw_segment, two_lambda_tau, rng);
        }

        let mut proposal = ProposedUpdate::new();
        proposal.add_position_modification(p0, t0..tot_slices, redraw_segment);

        trace!("\n{:?}", proposal);

        // // //

        let acceptance_ratio = action.potential_density_matrix_update(system, &proposal)
            * sum_weights
            / sum_weights_inverse;
        trace!("Acceptance ratio: {}", acceptance_ratio);

        // Apply Metropolis-Hastings acceptance criterion
        let proba = rng.random::<f64>();
        trace!("Drawn probability: {}", proba);
        if proba < acceptance_ratio {
            // Accept the update
            let worldlines_mut = system.path_mut();
            if let Some(modifications) = proposal.get_modifications(p0) {
                for (range, new_positions) in modifications {
                    worldlines_mut.set_positions(p0, range.start, range.end, new_positions);
                }
            }
            debug!("Move accepted");

            // Update the permutations
            worldlines_mut.set_following(p0, Some(tail));
            worldlines_mut.set_preceding(tail, Some(p0));
            worldlines_mut.set_preceding(tail_new, None); // New tail
            debug_assert!(
                worldlines_mut.worm_tail() == Some(tail_new),
                "Expected new tail {}, found {}",
                tail_new,
                worldlines_mut.worm_tail().unwrap()
            );

            // Bring the modified polymer to its standard form
            system.post_update_refactor(p0);

            Some(proposal.to_accepted_update())
        } else {
            // Reject the update
            debug!("Move rejected");
            None
        }
    }
}

// Inject the parameter tuning methods
impl_tunable_parameters!(Swap, (min_delta_t, usize), (max_delta_t, usize));
