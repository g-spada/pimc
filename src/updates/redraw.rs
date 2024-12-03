use super::levy_staging::levy_staging;
use super::monte_carlo_update::MonteCarloUpdate;
use super::proposed_update::ProposedUpdate;
use crate::path_state::traits::{
    WorldLineDimensions, WorldLinePermutationAccess, WorldLinePositionAccess,
};
use log::trace;
use ndarray::{s, Array2};
use std::cmp::min;

/// A Monte Carlo update that translates open and closed polymers.
pub struct Redraw<W> {
    min_delta_t: usize,
    max_delta_t: usize,
    two_lambda_tau: fn(&W, p: usize) -> f64,
    accept_count: usize,
    reject_count: usize,
}

impl<W> Redraw<W>
where
    W: WorldLineDimensions,
{
    /// Initializes `Redraw`.
    pub fn new(
        min_delta_t: usize,
        max_delta_t: usize,
        two_lambda_tau: fn(w: &W, _: usize) -> f64,
    ) -> Self {
        assert!(min_delta_t > 1, "`min_delta_t` must be greater than 1.");
        assert!(
            min_delta_t <= max_delta_t,
            "`min_delta_t`cannot be greater than `max_delta_t`"
        );
        Self {
            min_delta_t,
            max_delta_t,
            two_lambda_tau,
            accept_count: 0,
            reject_count: 0,
        }
    }
}

impl<W> MonteCarloUpdate<W, f64> for Redraw<W>
where
    W: WorldLineDimensions + WorldLinePositionAccess + WorldLinePermutationAccess,
{
    fn try_update<F>(
        &mut self,
        worldlines: &mut W,
        weight_function: F,
        rng: &mut impl rand::Rng,
    ) -> bool
    where
        F: Fn(&W, &ProposedUpdate<f64>) -> f64,
    {
        let mut proposal = ProposedUpdate::new();

        let n = worldlines.particles();
        let t = worldlines.time_slices();
        let d = worldlines.spatial_dimensions();

        // Randomly select an initial particle index
        let p0: usize = rng.gen_range(0..n);
        // Randomly select an initial time-slice
        let t0: usize = rng.gen_range(0..t - 1);
        // Randomly select the extent of the polymer to redraw. The segment is composed by `delta_t + 1` beads.
        // The first and last beads are kept fixed, the remaining `delta_t - 1` beads are redrawn.
        // The maximum extent is limited to t-2 to ensure two distinct fixed points in the staging
        // procedure (this limitation could in principle be relaxed to t-1 after modifying the
        // assertions on ProposedUpdate).
        let max_extent = min(self.max_delta_t, t - 2);
        let delta_t: usize = rng.gen_range(self.min_delta_t..=max_extent);

        // Get two_lambda_tau value for particle
        let two_lambda_tau = (self.two_lambda_tau)(&worldlines, p0);
        trace!(
            "Selected particle {}, initial slice {}, number of slices {}",
            p0,
            t0,
            delta_t
        );

        // Create an owned array and copy the initial bead (p0,t0)
        let mut redraw_segment = Array2::<f64>::zeros((delta_t + 1, d));
        redraw_segment
            .row_mut(0)
            .assign(&worldlines.position(p0, t0));

        if t0 + delta_t < t {
            // Segment to redraw is within the first polymer: copy the last bead
            redraw_segment
                .row_mut(delta_t)
                .assign(&worldlines.position(p0, t0 + delta_t));
            // Apply staging on the segment
            levy_staging(&mut redraw_segment, two_lambda_tau, rng);
            proposal.add_position_modification(p0, t0..(t0 + delta_t + 1), redraw_segment);
            trace!("Segment to redraw is contained in polymer {}. Staging beads from slice {} to slice {}", p0, t0, t0 + delta_t);
        } else {
            if let Some(p1) = worldlines.following(p0) {
                // Segment to redraw continues on next polymer p1.
                // Last bead on polymer p1 is at timeslice t1:
                let t1 = (t0 + delta_t) % (t - 1);
                trace!("Segment to redraw is split: using periodicity");
                trace!(
                    "Staging polymer {} from slice {} to slice {}",
                    p0,
                    t0,
                    t - 1
                );
                trace!("Staging polymer {} from slice {} to slice {}", p1, 0, t1);
                // Reconstruct a continuous path by removing the periodicity jumps induced by the space geometry:
                // * Compute the difference between the two images of the same bead
                let images_diff = &worldlines.position(p0, t - 1) - &worldlines.position(p1, 0);
                // * Insert image (without jumps) of the last bead
                redraw_segment
                    .row_mut(delta_t)
                    .assign(&(&worldlines.position(p1, t1) + &images_diff));
                // Apply staging on the segment
                levy_staging(&mut redraw_segment, two_lambda_tau, rng);
                proposal.add_position_modification(
                    p0,
                    t0..t,
                    redraw_segment.slice(s![..t - t0, ..]).to_owned(),
                );
                // We now remove images_diff from the p1 portion of the segment
                let mut selected_beads = redraw_segment.slice_mut(s![t - t0 - 1.., ..]);
                // Subtract the difference vector from all rows at once
                selected_beads -= &images_diff.broadcast(selected_beads.raw_dim()).unwrap();
                proposal.add_position_modification(
                    p1,
                    0..t1 + 1,
                    redraw_segment.slice(s![t - t0 - 1.., ..]).to_owned(),
                );
            } else {
                // Reached worm head, abort move
                trace!("Reached worm head, abort move");
                return false;
            }
        }
        trace!("Proposal update: {:?}", proposal);

        let acceptance_ratio = weight_function(worldlines, &proposal);
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
            true
        } else {
            // Reject the update
            self.reject_count += 1;
            false
        }
    }
}
