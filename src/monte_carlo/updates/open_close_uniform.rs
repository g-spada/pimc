use crate::action::traits::PotentialDensityMatrix;
use crate::impl_tunable_parameters;
use crate::monte_carlo::accepted_update::AcceptedUpdate;
use crate::monte_carlo::levy_staging::levy_staging;
use crate::monte_carlo::proposed_update::ProposedUpdate;
use crate::monte_carlo::traits::{MonteCarloStep, MonteCarloTunable};
use crate::path::sector::Sector;
use crate::path::traits::{
    WorldLineDimensions, WorldLinePermutationAccess, WorldLinePositionAccess, WorldLineStateEq,
    WorldLineWormAccess,
};
use crate::space::traits::Space;
use crate::system::traits::SystemAccess;
use log::{debug, trace};
use ndarray::Array2;

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
/// - `max_head_displacement`: The maximum allowed displacement of the worm's head.
///
/// # Implementation Details
/// - New head is proposed with a uniform distribution within the interval
///   [-max_head_displacement, max_head_displacement] for each spatial dimension.
///
/// # References
/// - Condens. Matter 2022, 7, 30, [<http://arxiv.org/abs/2203.00010>]
///   *Note*: Here we make use of a different definition for the open_close_constant that
///   incorporates the factor "tot_particles/volume".
#[derive(Debug, Clone, Copy)]
pub struct OpenCloseUniform {
    min_delta_t: usize,
    max_delta_t: usize,
    open_close_constant: f64,
    max_head_displacement: f64,
}

impl OpenCloseUniform {
    pub fn new(
        min_delta_t: usize,
        max_delta_t: usize,
        open_close_constant: f64,
        max_head_displacement: f64,
    ) -> Self {
        Self {
            min_delta_t,
            max_delta_t,
            open_close_constant,
            max_head_displacement,
        }
    }

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
            + WorldLineWormAccess,
        A: PotentialDensityMatrix,
        R: rand::Rng,
    {
        let worldlines = system.path();
        let tot_particles = worldlines.particles();
        let tot_slices = S::WorldLine::TIME_SLICES;
        let tot_directions = S::WorldLine::SPATIAL_DIMENSIONS;

        // Randomly select an initial particle index
        let p0: usize = rng.random_range(0..tot_particles);
        // Randomly select the extent of the polymer to redraw. The segment is composed by `delta_t + 1` beads.
        // The first bead is kept fixed, the remaining `delta_t` beads are redrawn.
        let delta_t: usize = rng.random_range(self.min_delta_t..=self.max_delta_t);
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

        // Create an owned array
        let mut redraw_segment = Array2::<f64>::zeros((delta_t + 1, tot_directions));
        // Copy the initial bead (p0,t0)
        redraw_segment
            .row_mut(0)
            .assign(&worldlines.position(p0, t0));
        // Copy the final (redundancy) bead (p0,tot_slices-1)
        redraw_segment
            .row_mut(delta_t)
            .assign(&worldlines.position(p0, tot_slices - 1));

        // Compute the old free-propagator weight
        let mut square_distance = 0.0;
        for i in 0..tot_directions {
            let dist_i = redraw_segment[[delta_t, i]] - redraw_segment[[0, i]];
            square_distance += dist_i * dist_i;
        }
        let rho_free_old = (-square_distance / (2.0 * two_lambda_tau * delta_t as f64)).exp();

        // Propose head (H) by shifting the position of the redundacy bead
        // in the range [-displacement_bound, displacement_bound] for each spatial direction
        let displacement_bound = f64::min(
            (two_lambda_tau * delta_t as f64).sqrt(),
            self.max_head_displacement,
        );
        for i in 0..tot_directions {
            redraw_segment[[delta_t, i]] +=
                rng.random_range(-displacement_bound..=displacement_bound);
        }

        // Compute the new free-propagator weight
        let mut square_distance = 0.0;
        for i in 0..tot_directions {
            let dist_i = redraw_segment[[delta_t, i]] - redraw_segment[[0, i]];
            square_distance += dist_i * dist_i;
        }
        let rho_free_new = (-square_distance / (2.0 * two_lambda_tau * delta_t as f64)).exp();

        // Compute the weight for the open move.
        // Note that we are exploiting the arbitrariness of `open_close_constant` to remove the
        // density factor
        let weight_open = self.open_close_constant
            * (2.0 * displacement_bound).powi(tot_directions as i32)
            * rho_free_new
            / rho_free_old;

        if delta_t > 1 {
            // Apply staging on the segment
            levy_staging(&mut redraw_segment, two_lambda_tau, rng);
        }

        let mut proposal = ProposedUpdate::new();
        proposal.add_position_modification(p0, t0..tot_slices, redraw_segment);
        trace!(
            "Open polymer {}. Beads from slice {} to slice {}",
            p0,
            t0,
            tot_slices
        );
        trace!("\n{:?}", proposal);

        let acceptance_ratio =
            action.potential_density_matrix_update(system, &proposal) * weight_open;
        trace!("Acceptance ratio: {}", acceptance_ratio);

        // Apply Metropolis-Hastings acceptance criterion
        let proba = rng.random::<f64>();
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
            debug!("Move accepted");

            // Update the permutations
            let tail = worldlines_mut.following(p0).unwrap();
            worldlines_mut.set_following(p0, None);
            worldlines_mut.set_preceding(tail, None);

            // Bring the modified polymer to its standard form
            system.post_update_refactor(p0);

            Some(proposal.to_accepted_update())
        } else {
            // Reject the update
            debug!("Move rejected");
            None
        }
    }

    fn close_polymer<S, A, R>(
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
        //let tot_particles = worldlines.particles();
        let tot_slices = S::WorldLine::TIME_SLICES;
        let tot_directions = S::WorldLine::SPATIAL_DIMENSIONS;

        // Randomly select the extent of the polymer to redraw. The segment is composed by `delta_t + 1` beads.
        // The first bead is kept fixed, the remaining `delta_t` beads are redrawn.
        let delta_t: usize = rng.random_range(self.min_delta_t..=self.max_delta_t);
        // Initial slice
        let t0 = tot_slices - delta_t - 1;

        let head = worldlines.worm_head().unwrap();
        let head_position = worldlines.position(head, tot_slices - 1);
        let tail = worldlines.worm_tail().unwrap();
        let tail_position = worldlines.position(tail, 0);

        // Reject the update if the internal state of head and tail differs
        if !worldlines.beads_state_eq(head, tot_slices - 1, tail, 0) {
            trace!("Internal quantum state of Head and Tail differs. Cannot close");
            return None;
        }

        // Get two_lambda_tau value for particle
        let two_lambda_tau = system.two_lambda_tau(head);
        // Check if TAIL and HEAD are close enough
        // i.e. within [-displacement_bound, displacement_bound] for each spatial direction
        let displacement_bound = f64::min(
            (two_lambda_tau * delta_t as f64).sqrt(),
            self.max_head_displacement,
        );

        let tail_head_distance = system.space().difference(tail_position, head_position);
        trace!(
            "Vector distance between head and tail (computed by space): {:#?}",
            tail_head_distance
        );

        if tail_head_distance
            .iter()
            .any(|&l| l.abs() > displacement_bound)
        {
            trace!(
                "Head ({:}) and Tail ({:}) are too far apart. Distance is {:} with bound {:}",
                head,
                tail,
                tail_head_distance,
                displacement_bound
            );
            return None;
        }

        // Create an owned array
        let mut redraw_segment = Array2::<f64>::zeros((delta_t + 1, tot_directions));

        // Copy the initial bead (head,t0)
        redraw_segment
            .row_mut(0)
            .assign(&worldlines.position(head, t0));

        // Set the final (redundancy) bead to the correct image of the tail
        redraw_segment
            .row_mut(delta_t)
            .assign(&(tail_head_distance + head_position));

        // Compute the old free-propagator weight
        let mut square_distance = 0.0;
        for i in 0..tot_directions {
            let dist_i = head_position[i] - redraw_segment[[0, i]];
            square_distance += dist_i * dist_i;
        }
        let rho_free_old = (-square_distance / (2.0 * two_lambda_tau * delta_t as f64)).exp();

        // Compute the new free-propagator weight
        let mut square_distance = 0.0;
        for i in 0..tot_directions {
            let dist_i = redraw_segment[[delta_t, i]] - redraw_segment[[0, i]];
            square_distance += dist_i * dist_i;
        }
        let rho_free_new = (-square_distance / (2.0 * two_lambda_tau * delta_t as f64)).exp();

        // Compute the prefactor for the acceptance probability
        let weight_close = (rho_free_new / rho_free_old)
            / (self.open_close_constant * (2.0 * displacement_bound).powi(tot_directions as i32));

        if delta_t > 1 {
            // Apply staging on the segment
            levy_staging(&mut redraw_segment, two_lambda_tau, rng);
        }

        let mut proposal = ProposedUpdate::new();
        proposal.add_position_modification(head, t0..tot_slices, redraw_segment);
        trace!(
            "Close worm head ({}). Beads from slice {} to slice {}",
            head,
            t0,
            tot_slices
        );
        trace!("\n{:?}", proposal);

        let acceptance_ratio =
            action.potential_density_matrix_update(system, &proposal) * weight_close;
        trace!("Acceptance ratio: {}", acceptance_ratio);

        // Apply Metropolis-Hastings acceptance criterion
        let proba = rng.random::<f64>();
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
            debug!("Move accepted");

            // Update the permutations
            worldlines_mut.set_following(head, Some(tail));
            worldlines_mut.set_preceding(tail, Some(head));

            // Bring the modified polymer to its standard form
            system.post_update_refactor(head);

            Some(proposal.to_accepted_update())
        } else {
            // Reject the update
            debug!("Move rejected");
            None
        }
    }
}

impl<S, A, R> MonteCarloStep<S, A, R> for OpenCloseUniform
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
    fn step(&mut self, system: &mut S, action: &A, rng: &mut R) -> Option<AcceptedUpdate> {
        if rng.random::<f64>() < 0.5 {
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

// Inject the parameter tuning methods
impl_tunable_parameters!(
    OpenCloseUniform,
    (min_delta_t, usize),
    (max_delta_t, usize),
    (open_close_constant, f64),
    (max_head_displacement, f64)
);

// Inject the stats tracking methods
