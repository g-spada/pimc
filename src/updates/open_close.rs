use super::accepted_update::AcceptedUpdate;
use super::levy_staging::levy_staging;
use super::monte_carlo_update::MonteCarloUpdate;
use super::proposed_update::ProposedUpdate;
use crate::action::traits::PotentialDensityMatrix;
use crate::path_state::sector::Sector;
use crate::path_state::traits::{
    WorldLineDimensions, WorldLinePermutationAccess, WorldLinePositionAccess, WorldLineStateEq,
    WorldLineWormAccess,
};
use crate::space::traits::Space;
use crate::system::traits::SystemAccess;
use log::{debug, trace};
use ndarray::Array2;
use rand_distr::{Distribution, Normal};
use std::f64::consts::PI;

/// A Monte Carlo update that opens/closes polymer cycles.
///
/// With the open update, a link between two beads is cut, creating two loose extremities and taking
/// the system from the Z-sector of closed paths to the G-sector with one worm. With the close
/// update, the two extremities of the worm are glued together, returning to the Z-sector.
///
/// # Fields
/// - `min_delta_t`: The minimum extent of the segment to redraw, in time slices.
/// - `max_delta_t`: The maximum extent of the segment to redraw, in time slices.
/// - `open_close_constant`: The constant that controls the relative simulation time spent in the
///    two sectors
/// - `accept_count`: Tracks the number of updates that have been accepted.
/// - `reject_count`: Tracks the number of updates that have been rejected.
///
/// # Implementation Details
/// - The new head is proposed with a free particle distribution. Detailed balance for systems with
///   periodic boundary conditions is ensured by rejecting updates with head too far away from the
///   corresponding image of the tail.
#[derive(Debug)]
pub struct OpenClose {
    pub min_delta_t: usize,
    pub max_delta_t: usize,
    pub open_close_constant: f64,
    pub accept_count: usize,
    pub reject_count: usize,
}

impl OpenClose {
    fn open_polymer<S, A, R>(
        &mut self,
        system: &mut S,
        action: &A,
        rng: &mut R,
    ) -> Option<AcceptedUpdate>
    where
        S: SystemAccess,
        S::WorldLine: WorldLineDimensions
            + WorldLinePositionAccess
            + WorldLinePermutationAccess
            + WorldLineStateEq
            + WorldLineWormAccess,
        A: PotentialDensityMatrix,
        R: rand::Rng,
    {
        let worldlines = system.path();
        let tot_particles = worldlines.particles();
        let tot_slices = S::WorldLine::TIME_SLICES;
        let tot_directions = S::WorldLine::SPATIAL_DIMENSIONS;

        // Randomly select an initial particle index
        let p0: usize = rng.gen_range(0..tot_particles);
        // Randomly select the extent of the polymer to redraw. The segment is composed by `delta_t + 1` beads.
        // The first bead is kept fixed, the remaining `delta_t` beads are redrawn.
        let delta_t: usize = rng.gen_range(self.min_delta_t..=self.max_delta_t);
        // Initial slice
        let t0 = tot_slices - delta_t - 1;

        // Get two_lambda_tau value for particle
        let two_lambda_tau = system.two_lambda_tau(p0);
        trace!(
            "Selected particle {}, initial slice {}, number of slices {}",
            p0,
            t0,
            delta_t
        );

        let r_pivot = worldlines.position(p0, t0);
        let r_periodicity = worldlines.position(p0, tot_slices - 1);

        // Create an Array2 containing the proposed new positions
        let mut redraw_segment = Array2::<f64>::zeros((delta_t + 1, tot_directions));

        // Copy the initial bead (p0,t0)
        redraw_segment.row_mut(0).assign(&r_pivot);

        // Propose HEAD bead according to the gaussian free-particle weight
        let sigma_sq = two_lambda_tau * delta_t as f64;
        let normal = Normal::new(0.0, sigma_sq.sqrt()).unwrap(); // Normal distribution (mean = 0, std_dev = sigma)
        redraw_segment.row_mut(delta_t).assign(&r_pivot);
        redraw_segment
            .row_mut(delta_t)
            .iter_mut()
            .for_each(|x| *x += normal.sample(rng));

        // Check if HEAD is within the correct tile, otherwise reject move
        let diff_euclidean = &redraw_segment.row(delta_t) - &r_periodicity;
        let diff_space = system
            .space()
            .difference(redraw_segment.row(delta_t), r_periodicity);
        if diff_euclidean != diff_space {
            debug!(
                "Proposed HEAD {:} is too far from previous periodicity bead (TAIL image).",
                redraw_segment.row(delta_t)
            );
            debug!("Inverse move does not exist.");
            debug!("Rejecting.");
            return None;
        }

        // Compute the free-propagator weight
        let distance_sq: f64 = (&r_pivot - &r_periodicity)
            .iter()
            .map(|x| x * x)
            .sum::<f64>();
        let rho_free = (-distance_sq / (2.0 * sigma_sq)).exp()
            / (2.0 * sigma_sq * PI).powi(tot_directions as i32 / 2);

        // Compute the weight for the open move. Note that we are exploiting the
        // arbitrariness of `open_close_constant` to remove the density factor
        let weight_open = self.open_close_constant / rho_free;

        if delta_t > 1 {
            // Apply staging on the segment
            levy_staging(&mut redraw_segment, two_lambda_tau, rng);
        }

        // Create update proposal
        let mut proposal = ProposedUpdate::new();
        proposal.add_position_modification(p0, t0..tot_slices, redraw_segment);
        trace!(
            "Open polymer {}. Beads from slice {} to slice {}",
            p0,
            t0,
            tot_slices
        );
        trace!("\n{:?}", proposal);

        // Compute acceptance ratio
        let acceptance_ratio =
            action.potential_density_matrix_update(system, &proposal) * weight_open;
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
            let tail = worldlines_mut.following(p0).unwrap();
            worldlines_mut.set_following(p0, None);
            worldlines_mut.set_preceding(tail, None);

            Some(proposal.to_accepted_update())
        } else {
            // Reject the update
            self.reject_count += 1;
            debug!("Move rejected");
            None
        }
    }

    fn close_polymer<S, A>(
        &mut self,
        system: &mut S,
        action: &A,
        rng: &mut impl rand::Rng,
    ) -> Option<AcceptedUpdate>
    where
        S: SystemAccess,
        S::WorldLine: WorldLineDimensions
            + WorldLinePositionAccess
            + WorldLinePermutationAccess
            + WorldLineStateEq
            + WorldLineWormAccess,
        A: PotentialDensityMatrix,
    {
        let worldlines = system.path();
        let tot_slices = S::WorldLine::TIME_SLICES;
        let tot_directions = S::WorldLine::SPATIAL_DIMENSIONS;

        let head = worldlines.worm_head().unwrap();
        let head_position = worldlines.position(head, tot_slices - 1);
        let tail = worldlines.worm_tail().unwrap();
        let tail_position = worldlines.position(tail, 0);

        // Reject the update if the internal state of head and tail differs
        if !worldlines.beads_state_eq(head, tot_slices - 1, tail, 0) {
            trace!("Internal quantum state of Head and Tail differs. Cannot close");
            return None;
        }

        // Randomly select the extent of the polymer to redraw. The segment is composed by `delta_t + 1` beads.
        // The first bead is kept fixed, the remaining `delta_t` beads are redrawn.
        let delta_t: usize = rng.gen_range(self.min_delta_t..=self.max_delta_t);
        // Initial slice
        let t0 = tot_slices - delta_t - 1;

        let pivot_position = worldlines.position(head, t0);

        // Get two_lambda_tau value for particle
        let two_lambda_tau = system.two_lambda_tau(head);

        // Find TAIL IMAGE that is the closest to HEAD
        let tail_head_distance = system.space().difference(tail_position, head_position);
        trace!(
            "Vector distance between head and tail (computed by space): {:#?}",
            tail_head_distance
        );

        // Create an Array2 containing the proposed new positions
        let mut redraw_segment = Array2::<f64>::zeros((delta_t + 1, tot_directions));

        // Copy the initial bead (head,t0)
        redraw_segment.row_mut(0).assign(&pivot_position);

        // Set the final (redundancy) bead to the correct image of the tail
        redraw_segment
            .row_mut(delta_t)
            .assign(&(tail_head_distance + head_position));

        // Compute the new free-propagator weight
        let distance_sq: f64 = (&redraw_segment.row(delta_t) - &pivot_position)
            .iter()
            .map(|x| x * x)
            .sum::<f64>();
        let sigma_sq = two_lambda_tau * delta_t as f64;
        let rho_free = (-distance_sq / (2.0 * sigma_sq)).exp()
            / (2.0 * sigma_sq * PI).powi(tot_directions as i32 / 2);

        // Compute the prefactor for the acceptance probability
        let weight_close = rho_free / self.open_close_constant;

        if delta_t > 1 {
            // Apply staging on the segment
            levy_staging(&mut redraw_segment, two_lambda_tau, rng);
        }

        // Create update proposal
        let mut proposal = ProposedUpdate::new();
        proposal.add_position_modification(head, t0..tot_slices, redraw_segment);
        trace!(
            "Close worm head ({}). Beads from slice {} to slice {}",
            head,
            t0,
            tot_slices
        );
        trace!("\n{:?}", proposal);

        // Compute acceptance ratio
        let acceptance_ratio =
            action.potential_density_matrix_update(system, &proposal) * weight_close;
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
            worldlines_mut.set_following(head, Some(tail));
            worldlines_mut.set_preceding(tail, Some(head));

            Some(proposal.to_accepted_update())
        } else {
            // Reject the update
            self.reject_count += 1;
            debug!("Move rejected");
            None
        }
    }
}

impl<S, A, R> MonteCarloUpdate<S, A, R> for OpenClose
where
    S: SystemAccess,
    S::WorldLine: WorldLineDimensions
        + WorldLinePositionAccess
        + WorldLinePermutationAccess
        + WorldLineStateEq
        + WorldLineWormAccess,
    A: PotentialDensityMatrix,
    R: rand::Rng,
{
    fn monte_carlo_update(
        &mut self,
        system: &mut S,
        action: &A,
        rng: &mut R,
    ) -> Option<AcceptedUpdate> {
        if rng.gen::<f64>() < 0.5 {
            // Try to open
            debug!("Trying open");
            if system.path().sector() == Sector::G {
                // Already open
                debug!("Already open. Nothing to update.");
                None
            } else {
                self.open_polymer(system, action, rng)
            }
        } else {
            // Try to close
            debug!("Trying close");
            if system.path().sector() == Sector::Z {
                // Already closed
                debug!("Already closed. Nothing to update.");
                None
            } else {
                self.close_polymer(system, action, rng)
            }
        }
    }
}
