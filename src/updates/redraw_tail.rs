use super::accepted_update::AcceptedUpdate;
use super::levy_staging::levy_staging;
use super::monte_carlo_update::MonteCarloUpdate;
use super::proposed_update::ProposedUpdate;
use crate::path_state::traits::{
    WorldLineDimensions, WorldLinePositionAccess, WorldLineWormAccess,
};
use log::{debug, trace};
use ndarray::Array2;
use rand_distr::{Distribution, Normal};

/// A Monte Carlo update that redraws the open segment containing the tail of the worm.
///
/// The `RedrawTail` struct implements a Monte Carlo update where the first (tail)
/// portion of the worm is randomly redrawn by sampling the new position of the
/// tail with the free propagator and then using the Levy staging algorithm for
/// the rest of the segment.
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
pub struct RedrawTail<F, W> {
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

    /// Tracks the number of updates that have been accepted.
    accept_count: usize,

    /// Tracks the number of updates that have been rejected.
    reject_count: usize,
}

impl<F, W> RedrawTail<F, W>
where
    F: Fn(&W, &ProposedUpdate<f64>) -> f64 + Send + Sync + 'static,
    W: WorldLineDimensions + WorldLinePositionAccess + WorldLineWormAccess,
{
    pub fn new(
        min_delta_t: usize,
        max_delta_t: usize,
        two_lambda_tau: fn(w: &W, _: usize) -> f64,
        weight_function: F,
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
            accept_count: 0,
            reject_count: 0,
        }
    }
}

impl<F, W> MonteCarloUpdate<W> for RedrawTail<F, W>
where
    F: Fn(&W, &ProposedUpdate<f64>) -> f64 + Send + Sync + 'static,
    W: WorldLineDimensions + WorldLinePositionAccess + WorldLineWormAccess,
{
    fn try_update(
        &mut self,
        worldlines: &mut W,
        rng: &mut impl rand::Rng,
    ) -> Option<AcceptedUpdate> {
        debug!("Trying update");

        // Find tail of the worm
        let tail = if let Some(tail) = worldlines.worm_tail() {
            tail
        } else {
            debug!("No tail to move");
            return None;
        };
        // Randomly select the extent of the polymer to redraw. The segment is composed by `delta_t + 1` beads.
        let delta_t: usize = rng.gen_range(self.min_delta_t..=self.max_delta_t);
        // Final slice
        let t0 = delta_t + 1;

        // Get two_lambda_tau value for particle
        let two_lambda_tau = (self.two_lambda_tau)(worldlines, tail);
        trace!("Redrawing worm tail ({}), up to slice {}", tail, t0,);

        // Create an owned array
        let mut redraw_segment = Array2::<f64>::zeros((t0, W::SPATIAL_DIMENSIONS));
        // Copy the final bead (tail, delta_t)
        redraw_segment
            .row_mut(delta_t)
            .assign(&worldlines.position(tail, delta_t));

        // Propose tail bead according to the gaussian free-particle weight
        let sigma = two_lambda_tau * delta_t as f64;
        let normal = Normal::new(0.0, sigma).unwrap(); // Normal distribution (mean = 0, std_dev = sigma)

        for i in 0..W::SPATIAL_DIMENSIONS {
            redraw_segment[[0, i]] = redraw_segment[[delta_t, i]] + normal.sample(rng);
        }

        if delta_t > 1 {
            // Apply staging on the rest of the segment
            levy_staging(&mut redraw_segment, two_lambda_tau, rng);
        }

        let mut proposal = ProposedUpdate::new();
        proposal.add_position_modification(tail, 0..t0, redraw_segment);

        trace!("\n{:?}", proposal);

        let acceptance_ratio = (self.weight_function)(worldlines, &proposal);
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
            Some(proposal.to_accepted_update())
        } else {
            // Reject the update
            self.reject_count += 1;
            debug!("Move rejected");
            None
        }
    }
}
