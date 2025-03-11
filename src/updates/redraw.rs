use super::accepted_update::AcceptedUpdate;
use super::levy_staging::levy_staging;
use super::monte_carlo_update::MonteCarloUpdate;
use super::proposed_update::ProposedUpdate;
use crate::action::traits::PotentialDensityMatrix;
use crate::path_state::traits::{
    WorldLineDimensions, WorldLinePermutationAccess, WorldLinePositionAccess,
};
use crate::system::traits::SystemAccess;
use log::{debug, trace};
use ndarray::{s, Array2};

/// A Monte Carlo update that redraws segments of open or closed polymers in a path integral simulation.
///
/// The `Redraw` struct implements a Monte Carlo move where segments of particle worldlines
/// are randomly selected and redrawn using the Levy staging algorithm. This move is
/// designed to improve sampling efficiency in simulations of quantum systems.
///
/// # Fields
/// - `min_delta_t`: The minimum length (in time slices) of a segment to redraw. Must be greater than 1.
/// - `max_delta_t`: The maximum length (in time slices) of a segment to redraw. Must be greater than or equal to `min_delta_t`.
/// - `accept_count`: Tracks the number of updates that have been accepted.
/// - `reject_count`: Tracks the number of updates that have been rejected.
#[derive(Debug)]
pub struct Redraw {
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

impl<S, A, R> MonteCarloUpdate<S, A, R> for Redraw
where
    S: SystemAccess,
    S::WorldLine: WorldLineDimensions + WorldLinePositionAccess + WorldLinePermutationAccess,
    A: PotentialDensityMatrix,
    R: rand::Rng,
{
    fn monte_carlo_update(
        &mut self,
        system: &mut S,
        action: &A,
        rng: &mut R,
    ) -> Option<AcceptedUpdate> {
        debug!("Trying update");
        let worldlines = system.path();
        let tot_particles = worldlines.particles();
        let tot_slices = S::WorldLine::TIME_SLICES;
        let tot_directions = S::WorldLine::SPATIAL_DIMENSIONS;

        // Randomly select an initial particle index
        let p0: usize = rng.gen_range(0..tot_particles);
        // Randomly select an initial time-slice
        let t0: usize = rng.gen_range(0..tot_slices - 1);
        // Randomly select the extent of the polymer to redraw. The segment is composed by `delta_t + 1` beads.
        // The first and last beads are kept fixed, the remaining `delta_t - 1` beads are redrawn.
        // The maximum extent is limited to t-2 to ensure two distinct fixed points in the staging
        // procedure (this limitation could in principle be relaxed to t-1 after modifying the
        // assertions on ProposedUpdate).
        let delta_t: usize = rng.gen_range(self.min_delta_t..=self.max_delta_t);

        let two_lambda_tau = system.two_lambda_tau(p0);

        trace!(
            "Selected particle {}, initial slice {}, number of slices {}",
            p0,
            t0,
            delta_t
        );

        // Create an owned array and copy the initial bead (p0,t0)
        let mut redraw_segment = Array2::<f64>::zeros((delta_t + 1, tot_directions));
        redraw_segment
            .row_mut(0)
            .assign(&worldlines.position(p0, t0));

        let mut proposal = ProposedUpdate::new();
        if t0 + delta_t < tot_slices {
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
            let t1 = (t0 + delta_t) % (tot_slices - 1);
            trace!("Segment to redraw is split: using periodicity");
            trace!(
                "Staging polymer {} from slice {} to slice {}",
                p0,
                t0,
                tot_slices - 1
            );
            trace!("Staging polymer {} from slice {} to slice {}", p1, 0, t1);
            // Reconstruct a continuous path by removing the periodicity jumps induced by the space geometry:
            // * Compute the difference between the two images of the same bead
            let images_diff =
                &worldlines.position(p0, tot_slices - 1) - &worldlines.position(p1, 0);
            // * Insert image (without jumps) of the last bead
            redraw_segment
                .row_mut(delta_t)
                .assign(&(&worldlines.position(p1, t1) + &images_diff));
            // Apply staging on the segment
            levy_staging(&mut redraw_segment, two_lambda_tau, rng);
            proposal.add_position_modification(
                p0,
                t0..tot_slices,
                redraw_segment.slice(s![..tot_slices - t0, ..]).to_owned(),
            );
            // We now remove images_diff from the p1 portion of the segment
            let mut selected_beads = redraw_segment.slice_mut(s![tot_slices - t0 - 1.., ..]);
            // Subtract the difference vector from all rows at once
            selected_beads -= &images_diff.broadcast(selected_beads.raw_dim()).unwrap();
            proposal.add_position_modification(
                p1,
                0..t1 + 1,
                redraw_segment
                    .slice(s![tot_slices - t0 - 1.., ..])
                    .to_owned(),
            );
        } else {
            // Reached worm head, abort move
            trace!("Reached worm head, abort move");
            return None;
        }
        trace!("\n{:?}", proposal);

        let acceptance_ratio = action.potential_density_matrix_update(system, &proposal);
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
            Some(proposal.to_accepted_update())
        } else {
            // Reject the update
            self.reject_count += 1;
            debug!("Move rejected");
            None
        }
    }
}
