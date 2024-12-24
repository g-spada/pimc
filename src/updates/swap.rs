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
use crate::space::traits::Space2;
use rand::distributions::WeightedIndex;
use rand_distr::Distribution;

/// A Monte Carlo update glues another polymer to the head of the worm, thus sampling permutations.
///
/// # Type Parameters
/// - `F`: A function type for computing the weight of a proposed update.
/// - `W`: The type representing the state of the particle worldlines, which must
///   implement the required traits for accessing and modifying particle positions.
///
/// # Fields
/// - `min_delta_t`: The minimum length (in time slices) of a segment to redraw. Must be greater than 1.
/// - `max_delta_t`: The maximum length (in time slices) of a segment to redraw. Must be greater than or equal to `min_delta_t`.
/// - `two_lambda_tau`: A function that computes a parameter required for the Levy staging algorithm,
///   based on the worldline state and the particle index.
/// - `weight_function`: A user-defined function that calculates the acceptance ratio for a proposed update.
/// - `accept_count`: Tracks the number of updates that have been accepted.
/// - `reject_count`: Tracks the number of updates that have been rejected.
///
/// # Usage
/// The `try_update` method performs the update and modifies the state of the worldlines if the
/// move is accepted.
///
/// # Panics
/// - If `min_delta_t` is less than or equal to 1.
/// - If `min_delta_t` is greater than `max_delta_t`.
///
/// # See Also
/// - `levy_staging`: The algorithm used for sampling the redrawn segment.
/// - `ProposedUpdate`: Used to manage and track the modifications made during the update.
///
pub struct Swap<'a, S: Space2, F, W> {
    /// The minimum extent of the segment to redraw, in time slices.
    /// Must be greater than 1.
    min_delta_t: usize,

    /// The maximum extent of the segment to redraw, in time slices.
    /// Must be greater than or equal to `min_delta_t`.
    max_delta_t: usize,

    /// A function that computes the `two_lambda_tau` parameter for the Levy staging algorithm
    /// based on the state of the worldlines and the particle index.
    two_lambda_tau: fn(&W, p: usize) -> f64,

    /// A user-defined function that calculates the acceptance ratio for the proposed update.
    /// This function takes the current state of the worldlines and the proposed update as input.
    weight_function: F,

    /// The underlying space
    space: &'a S,

    /// Tracks the number of updates that have been accepted.
    accept_count: usize,

    /// Tracks the number of updates that have been rejected.
    reject_count: usize,
}

impl<'a, S, F, W> Swap<'a, S, F, W>
where
    S: Space2,
    F: Fn(&W, &ProposedUpdate<f64>) -> f64 + Send + Sync + 'static,
    W: WorldLineDimensions
        + WorldLinePositionAccess
        + WorldLineWormAccess
        + WorldLinePermutationAccess
        + WorldLineBatchPositions,
{
    pub fn new(
        min_delta_t: usize,
        max_delta_t: usize,
        two_lambda_tau: fn(w: &W, _: usize) -> f64,
        weight_function: F,
        space: &'a S,
    ) -> Self {
        assert!(min_delta_t > 1, "`min_delta_t` must be greater than 1.");
        assert!(
            min_delta_t <= max_delta_t,
            "`min_delta_t`cannot be greater than `max_delta_t`"
        );
        assert!(
            max_delta_t < W::TIME_SLICES,
            "max_delta_t cannot be greater than W::TIME_SLICES -1"
        );
        Self {
            min_delta_t,
            max_delta_t,
            two_lambda_tau,
            weight_function,
            space,
            accept_count: 0,
            reject_count: 0,
        }
    }
}

impl<'a, S, F, W> MonteCarloUpdate<W> for Swap<'a, S, F, W>
where
    S: Space2,
    F: Fn(&W, &ProposedUpdate<f64>) -> f64 + Send + Sync + 'static,
    W: WorldLineDimensions
        + WorldLinePositionAccess
        + WorldLineWormAccess
        + WorldLinePermutationAccess
        + WorldLineBatchPositions,
{
    fn try_update(
        &mut self,
        worldlines: &mut W,
        rng: &mut impl rand::Rng,
    ) -> Option<AcceptedUpdate> {
        debug!("Trying update");

        // Find head of the worm
        let tail = if let Some(tail) = worldlines.worm_tail() {
            tail
        } else {
            debug!("No worm tail present");
            return None;
        };
        let head = worldlines.worm_head().unwrap();
        //let r_head_boxed = self.space.base_image(worldlines.position(head, W::TIME_SLICES - 1 ) );

        // Randomly select the extent of the polymer to redraw
        let delta_t: usize = rng.gen_range(self.min_delta_t..=self.max_delta_t);

        // Pivot slice
        let t0 = W::TIME_SLICES - delta_t - 1;

        // Get two_lambda_tau value for particle
        // TODO: THINK ABOUT THIS
        let two_lambda_tau = (self.two_lambda_tau)(worldlines, tail);

        // Nearest images
        let nearest_images = self.space.differences_from_reference(
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
        let nearest_images_p0 = self.space.differences_from_reference(
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

        let mut redraw_segment = Array2::<f64>::zeros((delta_t + 1, W::SPATIAL_DIMENSIONS));
        // Copy the initial bead (p0,t0)
        redraw_segment
            .row_mut(0)
            .assign(&worldlines.position(p0, t0));

        // Final bead: glue to previous tail
        redraw_segment
            .row_mut(delta_t)
            .assign(&(&worldlines.position(p0, t0) - &nearest_images.row(p0)));

        // We now redraw the segment t0..W::TIME_SLICES with levy
        if delta_t > 1 {
            // Apply staging on the rest of the segment
            levy_staging(&mut redraw_segment, two_lambda_tau, rng);
        }

        let mut proposal = ProposedUpdate::new();
        proposal.add_position_modification(p0, t0..W::TIME_SLICES, redraw_segment);

        trace!("\n{:?}", proposal);

        // // //

        let acceptance_ratio =
            (self.weight_function)(worldlines, &proposal) * sum_weights / sum_weights_p0;
        trace!("Acceptance ratio: {}", acceptance_ratio);

        // Apply Metropolis-Hastings acceptance criterion
        let proba = rng.gen::<f64>();
        trace!("Drawn probability: {}", proba);
        if proba < acceptance_ratio {
            // Accept the update
            for particle in proposal.get_modified_particles() {
                if let Some(modifications) = proposal.get_modifications(particle) {
                    for (range, new_positions) in modifications {
                        worldlines.set_positions(particle, range.start, range.end, new_positions);
                    }
                }
            }
            self.accept_count += 1;
            debug!("Move accepted");

            // Update the permutations
            let broken_link = worldlines.following(p0).unwrap(); // Can not fail: p0 is not head
            worldlines.set_following(p0, Some(tail));
            worldlines.set_preceding(tail, Some(p0));
            worldlines.set_preceding(broken_link, None);
            debug_assert!(
                worldlines.worm_tail() == Some(broken_link),
                "Expected new tail {}, found {}",
                broken_link,
                worldlines.worm_tail().unwrap()
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
