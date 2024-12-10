use super::accepted_update::AcceptedUpdate;
use super::levy_staging::levy_staging;
use super::monte_carlo_update::MonteCarloUpdate;
use super::proposed_update::ProposedUpdate;
use crate::path_state::traits::{
    WorldLineDimensions, WorldLinePermutationAccess, WorldLinePositionAccess,
};
use log::{debug, trace};
use ndarray::{s, Array2};

/// A Monte Carlo update that redraws segments of open or closed polymers in a path integral simulation.
///
/// The `Redraw` struct implements a Monte Carlo move where segments of particle worldlines
/// are randomly selected and redrawn using the Levy staging algorithm. This move is
/// designed to improve sampling efficiency in simulations of quantum systems.
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
/// The `Redraw` struct is typically used in path integral Monte Carlo simulations.
/// It selects a random particle and a segment of its worldline, applies Levy staging to redraw
/// the segment, and evaluates the move using the Metropolis-Hastings criterion.
///
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
pub struct Redraw<F, W> {
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

impl<F, W> Redraw<F, W>
where
    F: Fn(&W, &ProposedUpdate<f64>) -> f64 + Send + Sync + 'static,
    W: WorldLineDimensions + WorldLinePositionAccess + WorldLinePermutationAccess,
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
            max_delta_t < W::TIME_SLICES - 1,
            "max_delta_t cannot be greater than W::TIME_SLICES - 2"
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

impl<F, W> MonteCarloUpdate<W> for Redraw<F, W>
where
    F: Fn(&W, &ProposedUpdate<f64>) -> f64 + Send + Sync + 'static,
    W: WorldLineDimensions + WorldLinePositionAccess + WorldLinePermutationAccess,
{
    fn try_update(
        &mut self,
        worldlines: &mut W,
        rng: &mut impl rand::Rng,
    ) -> Option<AcceptedUpdate> {
        debug!("Trying update");
        let mut proposal = ProposedUpdate::new();

        let tot_particles = worldlines.particles();

        // Randomly select an initial particle index
        let p0: usize = rng.gen_range(0..tot_particles);
        // Randomly select an initial time-slice
        let t0: usize = rng.gen_range(0..W::TIME_SLICES - 1);
        // Randomly select the extent of the polymer to redraw. The segment is composed by `delta_t + 1` beads.
        // The first and last beads are kept fixed, the remaining `delta_t - 1` beads are redrawn.
        // The maximum extent is limited to t-2 to ensure two distinct fixed points in the staging
        // procedure (this limitation could in principle be relaxed to t-1 after modifying the
        // assertions on ProposedUpdate).
        let delta_t: usize = rng.gen_range(self.min_delta_t..=self.max_delta_t);

        // Get two_lambda_tau value for particle
        let two_lambda_tau = (self.two_lambda_tau)(worldlines, p0);
        trace!(
            "Selected particle {}, initial slice {}, number of slices {}",
            p0,
            t0,
            delta_t
        );

        // Create an owned array and copy the initial bead (p0,t0)
        let mut redraw_segment = Array2::<f64>::zeros((delta_t + 1, W::SPATIAL_DIMENSIONS));
        redraw_segment
            .row_mut(0)
            .assign(&worldlines.position(p0, t0));

        if t0 + delta_t < W::TIME_SLICES {
            // Segment to redraw is within the first polymer: copy the last bead
            redraw_segment
                .row_mut(delta_t)
                .assign(&worldlines.position(p0, t0 + delta_t));
            // Apply staging on the segment
            levy_staging(&mut redraw_segment, two_lambda_tau, rng);
            proposal.add_position_modification(p0, t0..(t0 + delta_t + 1), redraw_segment);
            trace!("Segment to redraw is contained in polymer {}. Staging beads from slice {} to slice {}", p0, t0, t0 + delta_t);
        } else if let Some(p1) = worldlines.following(p0) {
            // Segment to redraw continues on next polymer p1.
            // Last bead on polymer p1 is at timeslice t1:
            let t1 = (t0 + delta_t) % (W::TIME_SLICES - 1);
            trace!("Segment to redraw is split: using periodicity");
            trace!(
                "Staging polymer {} from slice {} to slice {}",
                p0,
                t0,
                W::TIME_SLICES - 1
            );
            trace!("Staging polymer {} from slice {} to slice {}", p1, 0, t1);
            // Reconstruct a continuous path by removing the periodicity jumps induced by the space geometry:
            // * Compute the difference between the two images of the same bead
            let images_diff =
                &worldlines.position(p0, W::TIME_SLICES - 1) - &worldlines.position(p1, 0);
            // * Insert image (without jumps) of the last bead
            redraw_segment
                .row_mut(delta_t)
                .assign(&(&worldlines.position(p1, t1) + &images_diff));
            // Apply staging on the segment
            levy_staging(&mut redraw_segment, two_lambda_tau, rng);
            proposal.add_position_modification(
                p0,
                t0..W::TIME_SLICES,
                redraw_segment
                    .slice(s![..W::TIME_SLICES - t0, ..])
                    .to_owned(),
            );
            // We now remove images_diff from the p1 portion of the segment
            let mut selected_beads = redraw_segment.slice_mut(s![W::TIME_SLICES - t0 - 1.., ..]);
            // Subtract the difference vector from all rows at once
            selected_beads -= &images_diff.broadcast(selected_beads.raw_dim()).unwrap();
            proposal.add_position_modification(
                p1,
                0..t1 + 1,
                redraw_segment
                    .slice(s![W::TIME_SLICES - t0 - 1.., ..])
                    .to_owned(),
            );
        } else {
            // Reached worm head, abort move
            trace!("Reached worm head, abort move");
            return None;
        }
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
