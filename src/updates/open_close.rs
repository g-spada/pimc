use super::accepted_update::AcceptedUpdate;
use super::levy_staging::levy_staging;
use super::monte_carlo_update::MonteCarloUpdate;
use super::proposed_update::ProposedUpdate;
use crate::path_state::sector::Sector;
use crate::path_state::traits::{
    WorldLineDimensions, WorldLinePermutationAccess, WorldLinePositionAccess, WorldLineWormAccess,
};
use crate::space::traits::Space2;
use log::{debug, trace};
use ndarray::Array2;

/// A Monte Carlo update that opens/closes polymer cycles.
///
/// With the open update, a link between two beads is cut, creating two loose extremities and taking
/// the system from the Z-sector of closed paths to the G-sector with one worm. With the close
/// update, the two extremities of the worm are glued together, returning to the Z-sector.
/// Open and close updates are the opposite of each other and their transition probability must be
/// properly implemented to balance the Markov Chain. In this implementation, we propose to open or
/// close the worm at the same rate, independently of the sector the simulation is in, and,
/// only afterwards, we abort the move if either the open move is called within the G-sector
/// or the close move is called within the Z-sector.
///
/// # Type Parameters
/// - `F`: A function type for computing the weight of a proposed update.
/// - `W`: The type representing the state of the particle worldlines, which must
///   implement the required traits for modifying particle positions, permutations and
///
/// # Fields
/// - `min_delta_t`: The minimum extent of the segment to redraw, in time slices.
/// - `max_delta_t`: The maximum extent of the segment to redraw, in time slices.
/// - `open_close_constant`: The constant that controls the relative simulation time spent in the
///    two sectors
/// - `two_lambda_tau`: A function that computes a parameter required for the Levy staging algorithm,
///   based on the worldline state and the particle index.
/// - `weight_function`: A user-defined function that calculates the acceptance ratio for a proposed update.
/// - `accept_count`: Tracks the number of updates that have been accepted.
/// - `reject_count`: Tracks the number of updates that have been rejected.
///
/// # Usage
/// The `OpenClose` struct is typically used in path integral Monte Carlo simulations.
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
pub struct OpenClose<'a, S, F, W>
where
    S: Space2,
    F: Fn(&W, &ProposedUpdate<f64>) -> f64 + Send + Sync + 'static,
    W: WorldLineDimensions
        + WorldLinePositionAccess
        + WorldLinePermutationAccess
        + WorldLineWormAccess,
{
    min_delta_t: usize,
    max_delta_t: usize,
    open_close_constant: f64,
    max_head_displacement: f64,
    space: &'a S,
    two_lambda_tau: fn(&W, p: usize) -> f64,
    weight_function: F,
    accept_count: usize,
    reject_count: usize,
}

impl<'a, S, F, W> OpenClose<'a, S, F, W>
where
    S: Space2,
    F: Fn(&W, &ProposedUpdate<f64>) -> f64 + Send + Sync + 'static,
    W: WorldLineDimensions
        + WorldLinePositionAccess
        + WorldLinePermutationAccess
        + WorldLineWormAccess,
{
    pub fn new(
        min_delta_t: usize,
        max_delta_t: usize,
        open_close_constant: f64,
        max_head_displacement: f64,
        space: &'a S,
        two_lambda_tau: fn(w: &W, _: usize) -> f64,
        weight_function: F,
    ) -> Self {
        assert!(min_delta_t > 0, "`min_delta_t` must be greater than 0.");
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
            open_close_constant,
            max_head_displacement,
            space,
            two_lambda_tau,
            weight_function,
            accept_count: 0,
            reject_count: 0,
        }
    }

    fn open_polymer(
        &mut self,
        worldlines: &mut W,
        rng: &mut impl rand::Rng,
    ) -> Option<AcceptedUpdate> {
        let tot_particles = worldlines.particles();

        // Randomly select an initial particle index
        let p0: usize = rng.gen_range(0..tot_particles);
        // Randomly select the extent of the polymer to redraw. The segment is composed by `delta_t + 1` beads.
        // The first bead is kept fixed, the remaining `delta_t` beads are redrawn.
        let delta_t: usize = rng.gen_range(self.min_delta_t..=self.max_delta_t);
        // Initial slice
        let t0 = W::TIME_SLICES - delta_t - 1;

        // Get two_lambda_tau value for particle
        let two_lambda_tau = (self.two_lambda_tau)(worldlines, p0);
        trace!(
            "Selected particle {}, initial slice {}, number of slices {}",
            p0,
            t0,
            delta_t
        );

        // Create an owned array
        let mut redraw_segment = Array2::<f64>::zeros((delta_t + 1, W::SPATIAL_DIMENSIONS));
        // Copy the initial bead (p0,t0)
        redraw_segment
            .row_mut(0)
            .assign(&worldlines.position(p0, t0));
        // Copy the final (redundancy) bead (p0,W::TIME_SLICES-1)
        redraw_segment
            .row_mut(delta_t)
            .assign(&worldlines.position(p0, W::TIME_SLICES - 1));

        // Compute the old free-propagator weight
        let mut square_distance = 0.0;
        for i in 0..W::SPATIAL_DIMENSIONS {
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
        for i in 0..W::SPATIAL_DIMENSIONS {
            redraw_segment[[delta_t, i]] += rng.gen_range(-displacement_bound..=displacement_bound);
        }

        // Compute the new free-propagator weight
        let mut square_distance = 0.0;
        for i in 0..W::SPATIAL_DIMENSIONS {
            let dist_i = redraw_segment[[delta_t, i]] - redraw_segment[[0, i]];
            square_distance += dist_i * dist_i;
        }
        let rho_free_new = (-square_distance / (2.0 * two_lambda_tau * delta_t as f64)).exp();

        let weight_open = self.open_close_constant * tot_particles as f64 / self.space.volume()
            * (2.0 * displacement_bound).powi(W::SPATIAL_DIMENSIONS as i32)
            * rho_free_new
            / rho_free_old;

        if delta_t > 1 {
            // Apply staging on the segment
            levy_staging(&mut redraw_segment, two_lambda_tau, rng);
        }

        let mut proposal = ProposedUpdate::new();
        proposal.add_position_modification(p0, t0..W::TIME_SLICES, redraw_segment);
        trace!(
            "Open polymer {}. Beads from slice {} to slice {}",
            p0,
            t0,
            W::TIME_SLICES
        );
        trace!("\n{:?}", proposal);

        let acceptance_ratio = weight_open * (self.weight_function)(worldlines, &proposal);
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
            let tail = worldlines.following(p0).unwrap();
            worldlines.set_following(p0, None);
            worldlines.set_preceding(tail, None);

            Some(proposal.to_accepted_update())
        } else {
            // Reject the update
            self.reject_count += 1;
            debug!("Move rejected");
            None
        }
    }

    fn close_polymer(
        &mut self,
        worldlines: &mut W,
        rng: &mut impl rand::Rng,
    ) -> Option<AcceptedUpdate> {
        let tot_particles = worldlines.particles();

        // Randomly select the extent of the polymer to redraw. The segment is composed by `delta_t + 1` beads.
        // The first bead is kept fixed, the remaining `delta_t` beads are redrawn.
        let delta_t: usize = rng.gen_range(self.min_delta_t..=self.max_delta_t);
        // Initial slice
        let t0 = W::TIME_SLICES - delta_t - 1;

        let head = worldlines.worm_head().unwrap();
        let head_position = worldlines.position(head, W::TIME_SLICES - 1);
        let tail = worldlines.worm_tail().unwrap();
        let tail_position = worldlines.position(tail, 0);
        // Get two_lambda_tau value for particle
        let two_lambda_tau = (self.two_lambda_tau)(worldlines, head);
        // Check if TAIL and HEAD are close enough
        // i.e. within [-displacement_bound, displacement_bound] for each spatial direction
        let displacement_bound = f64::min(
            (two_lambda_tau * delta_t as f64).sqrt(),
            self.max_head_displacement,
        );

        let tail_head_distance = self.space.difference(tail_position, head_position);
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
        let mut redraw_segment = Array2::<f64>::zeros((delta_t + 1, W::SPATIAL_DIMENSIONS));

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
        for i in 0..W::SPATIAL_DIMENSIONS {
            let dist_i = head_position[i] - redraw_segment[[0, i]];
            square_distance += dist_i * dist_i;
        }
        let rho_free_old = (-square_distance / (2.0 * two_lambda_tau * delta_t as f64)).exp();

        // Compute the new free-propagator weight
        let mut square_distance = 0.0;
        for i in 0..W::SPATIAL_DIMENSIONS {
            let dist_i = redraw_segment[[delta_t, i]] - redraw_segment[[0, i]];
            square_distance += dist_i * dist_i;
        }
        let rho_free_new = (-square_distance / (2.0 * two_lambda_tau * delta_t as f64)).exp();

        // Compute the prefactor for the acceptance probability
        let weight_close = self.space.volume()
            / (self.open_close_constant
                * tot_particles as f64
                * (2.0 * displacement_bound).powi(W::SPATIAL_DIMENSIONS as i32))
            * rho_free_new
            / rho_free_old;

        if delta_t > 1 {
            // Apply staging on the segment
            levy_staging(&mut redraw_segment, two_lambda_tau, rng);
        }

        let mut proposal = ProposedUpdate::new();
        proposal.add_position_modification(head, t0..W::TIME_SLICES, redraw_segment);
        trace!(
            "Close worm head ({}). Beads from slice {} to slice {}",
            head,
            t0,
            W::TIME_SLICES
        );
        trace!("\n{:?}", proposal);

        let acceptance_ratio = weight_close * (self.weight_function)(worldlines, &proposal);
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
            worldlines.set_following(head, Some(tail));
            worldlines.set_preceding(tail, Some(head));

            Some(proposal.to_accepted_update())
        } else {
            // Reject the update
            self.reject_count += 1;
            debug!("Move rejected");
            None
        }
    }
}

impl<'a, S, F, W> MonteCarloUpdate<W> for OpenClose<'a, S, F, W>
where
    S: Space2,
    F: Fn(&W, &ProposedUpdate<f64>) -> f64 + Send + Sync + 'static,
    W: WorldLineDimensions
        + WorldLinePositionAccess
        + WorldLinePermutationAccess
        + WorldLineWormAccess,
{
    fn try_update(
        &mut self,
        worldlines: &mut W,
        rng: &mut impl rand::Rng,
    ) -> Option<AcceptedUpdate> {
        if rng.gen::<f64>() < 0.5 {
            // Try to open
            debug!("Trying open");
            if worldlines.sector() == Sector::G {
                // Already open
                debug!("Already open. Nothing to update.");
                None
            } else {
                self.open_polymer(worldlines, rng)
            }
        } else {
            // Try to close
            debug!("Trying close");
            if worldlines.sector() == Sector::Z {
                // Already closed
                debug!("Already closed. Nothing to update.");
                None
            } else {
                self.close_polymer(worldlines, rng)
            }
        }
    }
}
