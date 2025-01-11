use super::accepted_update::AcceptedUpdate;
use super::levy_staging::levy_staging;
use super::monte_carlo_update::MonteCarloUpdate;
use super::proposed_update::ProposedUpdate;
use crate::path_state::traits::{
    WorldLineBatchPositions, WorldLineDimensions, WorldLinePermutationAccess,
    WorldLinePositionAccess, WorldLineWormAccess,
};
use log::{debug, trace};
use ndarray::{Array1, Array2};
//use ndarray::linalg::norm_l2;
use crate::action::traits::PotentialDensityMatrix;
//use crate::space::traits::Space;
use crate::space::traits::Space;
use crate::system::traits::SystemAccess;
use rand::distributions::WeightedIndex;
use rand_distr::Distribution;

/// A Monte Carlo update glues another polymer to the head of the worm, thus sampling permutations.
///
/// # Fields
/// - `min_delta_t`: The minimum length (in time slices) of a segment to redraw. Must be greater than 1.
/// - `max_delta_t`: The maximum length (in time slices) of a segment to redraw. Must be greater than or equal to `min_delta_t`.
/// - `accept_count`: Tracks the number of updates that have been accepted.
/// - `reject_count`: Tracks the number of updates that have been rejected.
pub struct Swap {
    /// The minimum extent of the segment to redraw, in time slices.
    /// Must be greater than 1.
    pub min_delta_t: usize,

    /// The maximum extent of the segment to redraw, in time slices.
    /// Must be greater than or equal to `min_delta_t`.
    pub max_delta_t: usize,

    /// Tracks the number of updates that have been accepted.
    pub accept_count: usize,

    /// Tracks the number of updates that have been rejected.
    pub reject_count: usize,
}

impl<S, A> MonteCarloUpdate<S, A> for Swap
where
    S: SystemAccess,
    S::WorldLine: WorldLineDimensions
        + WorldLinePositionAccess
        + WorldLineWormAccess
        + WorldLinePermutationAccess
        + WorldLineBatchPositions,
    A: PotentialDensityMatrix,
{
    fn monte_carlo_update(
        &mut self,
        system: &mut S,
        action: &A,
        rng: &mut impl rand::Rng,
    ) -> Option<AcceptedUpdate> {
        debug!("Trying update");

        let worldlines = system.path();
        //let tot_particles = worldlines.particles();
        let tot_slices = S::WorldLine::TIME_SLICES;
        let tot_directions = S::WorldLine::SPATIAL_DIMENSIONS;
        // Find head of the worm
        let tail = if let Some(tail) = worldlines.worm_tail() {
            tail
        } else {
            debug!("No worm tail present");
            return None;
        };
        let head = worldlines.worm_head().unwrap();

        // Randomly select the extent of the polymer to redraw
        let delta_t: usize = rng.gen_range(self.min_delta_t..=self.max_delta_t);

        // Pivot slice
        let t0 = tot_slices - delta_t - 1;

        // Get two_lambda_tau value for particle
        let two_lambda_tau = system.two_lambda_tau(tail);

        // Nearest images
        let nearest_images = system.space().differences_from_reference(
            worldlines.positions_at_time_slice(t0),
            worldlines.position(tail, 0),
        );
        trace!("Distances at pivot slice T{}:\n{:?}", t0, nearest_images);

        // Compute the weights
        let mut weights: Array1<f64> = nearest_images
            .outer_iter()
            .map(|row| {
                let distance_sq = row.iter().map(|x| x * x).sum::<f64>();
                //let distance_sq = norm_l2(&row); // Compute Euclidean modulus (L2 norm)
                (-distance_sq / (2.0 * two_lambda_tau * delta_t as f64)).exp()
            })
            .collect();
        weights[head] = 0.0;
        trace!("Weights: {:?}", weights);
        let sum_weights = weights.sum();

        // Select the particle by tower sampling from the discrete probability distribution given by
        // weights
        let p0 = WeightedIndex::new(&weights).unwrap().sample(rng);
        debug_assert!(p0 != head, "Cannot swap with head");
        trace!("Selected P{}, from pivot slice T{}", p0, t0,);

        // Compute weight from inverse move
        // Nearest images of the selected particle
        let nearest_images_p0 = system.space().differences_from_reference(
            worldlines.positions_at_time_slice(t0),
            worldlines.position(p0, 0),
        );
        weights = nearest_images_p0
            .outer_iter()
            .map(|row| {
                let distance_sq = row.iter().map(|x| x * x).sum::<f64>();
                //let distance_sq = norm_l2(&row); // Compute Euclidean modulus (L2 norm)
                (-distance_sq / (2.0 * two_lambda_tau * delta_t as f64)).exp()
            })
            .collect();
        weights[head] = 0.0;
        let sum_weights_p0 = weights.sum();

        let mut redraw_segment = Array2::<f64>::zeros((delta_t + 1, tot_directions));
        // Copy the initial bead (p0,t0)
        redraw_segment
            .row_mut(0)
            .assign(&worldlines.position(p0, t0));

        // Final bead: glue to previous tail
        redraw_segment
            .row_mut(delta_t)
            .assign(&(&worldlines.position(p0, t0) - &nearest_images.row(p0)));

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
            / sum_weights_p0;
        trace!("Acceptance ratio: {}", acceptance_ratio);

        // Apply Metropolis-Hastings acceptance criterion
        let proba = rng.gen::<f64>();
        trace!("Drawn probability: {}", proba);
        if proba < acceptance_ratio {
            // Accept the update
            let worldlines_mut = system.path_mut();
            for particle in proposal.get_modified_particles() {
                if let Some(modifications) = proposal.get_modifications(particle) {
                    for (range, new_positions) in modifications {
                        worldlines_mut.set_positions(
                            particle,
                            range.start,
                            range.end,
                            new_positions,
                        );
                    }
                }
            }
            self.accept_count += 1;
            debug!("Move accepted");

            // Update the permutations
            let broken_link = worldlines_mut.following(p0).unwrap(); // Can not fail: p0 is not head
            worldlines_mut.set_following(p0, Some(tail));
            worldlines_mut.set_preceding(tail, Some(p0));
            worldlines_mut.set_preceding(broken_link, None);
            debug_assert!(
                worldlines_mut.worm_tail() == Some(broken_link),
                "Expected new tail {}, found {}",
                broken_link,
                worldlines_mut.worm_tail().unwrap()
            );

            Some(proposal.to_accepted_update())
        } else {
            // Reject the update
            self.reject_count += 1;
            debug!("Move rejected");
            None
        }
    }
}
