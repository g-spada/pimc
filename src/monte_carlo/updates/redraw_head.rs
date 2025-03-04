use crate::action::traits::PotentialDensityMatrix;
use crate::impl_tunable_parameters;
use crate::monte_carlo::accepted_update::AcceptedUpdate;
use crate::monte_carlo::levy_staging::levy_staging;
use crate::monte_carlo::proposed_update::ProposedUpdate;
use crate::monte_carlo::traits::{MonteCarloStep, MonteCarloTunable};
use crate::path::traits::{WorldLineDimensions, WorldLinePositionAccess, WorldLineWormAccess};
use crate::system::traits::SystemAccess;
use log::{debug, trace};
use ndarray::Array2;
use rand_distr::{Distribution, Normal};

/// A Monte Carlo update that redraws the open segment containing the head of the worm.
///
/// The `RedrawHead` struct implements a Monte Carlo update where the last (head)
/// portion of the worm is randomly redrawn by sampling the new position of the
/// head with the free propagator and then using the Levy staging algorithm for
/// the rest of the segment.
///
/// # Fields
/// - `min_delta_t`: The minimum length (in time slices) of a segment to redraw. Must be greater than 1.
/// - `max_delta_t`: The maximum length (in time slices) of a segment to redraw. Must be greater than or equal to `min_delta_t`.
#[derive(Debug, Clone, Copy)]
pub struct RedrawHead {
    /// The minimum extent of the segment to redraw, in time slices.
    /// Must be greater than 1.
    min_delta_t: usize,

    /// The maximum extent of the segment to redraw, in time slices.
    /// Must be greater than or equal to `min_delta_t`.
    max_delta_t: usize,
}

impl RedrawHead {
    pub fn new(min_delta_t: usize, max_delta_t: usize) -> Self {
        Self {
            min_delta_t,
            max_delta_t,
        }
    }
}

impl<S, A, R> MonteCarloStep<S, A, R> for RedrawHead
where
    S: SystemAccess,
    S::WorldLine: WorldLineDimensions + WorldLinePositionAccess + WorldLineWormAccess,
    A: PotentialDensityMatrix,
    R: rand::Rng,
{
    fn step(&mut self, system: &mut S, action: &A, rng: &mut R) -> Option<AcceptedUpdate> {
        debug!("Trying update");
        let worldlines = system.path();
        let tot_slices = S::WorldLine::TIME_SLICES;
        let tot_directions = S::WorldLine::SPATIAL_DIMENSIONS;

        // Find head of the worm
        let head = if let Some(head) = worldlines.worm_head() {
            head
        } else {
            debug!("No head to move");
            return None;
        };
        // Randomly select the extent of the polymer to redraw. The segment is composed by `delta_t + 1` beads.
        let delta_t: usize = rng.random_range(self.min_delta_t..=self.max_delta_t);
        // Initial slice
        let t0 = tot_slices - delta_t - 1;

        // Get two_lambda_tau value for particle
        let two_lambda_tau = system.two_lambda_tau(head);
        trace!("Redrawing worm head ({}), from initial slice {}", head, t0,);

        // Create an owned array
        let mut redraw_segment = Array2::<f64>::zeros((delta_t + 1, tot_directions));
        // Copy the initial bead (head,t0)
        redraw_segment
            .row_mut(0)
            .assign(&worldlines.position(head, t0));

        // Propose head bead according to the gaussian free-particle weight
        let sigma = (two_lambda_tau * delta_t as f64).sqrt();
        let normal = Normal::new(0.0, sigma).unwrap(); // Normal distribution (mean = 0, std_dev = sigma)

        for i in 0..tot_directions {
            redraw_segment[[delta_t, i]] = redraw_segment[[0, i]] + normal.sample(rng);
        }

        if delta_t > 1 {
            // Apply staging on the rest of the segment
            levy_staging(&mut redraw_segment, two_lambda_tau, rng);
        }
        let mut proposal = ProposedUpdate::new();
        proposal.add_position_modification(head, t0..tot_slices, redraw_segment);

        trace!("\n{:?}", proposal);

        let acceptance_ratio = action.potential_density_matrix_update(system, &proposal);
        trace!("Acceptance ratio: {}", acceptance_ratio);

        // Apply Metropolis-Hastings acceptance criterion
        let proba = rng.random::<f64>();
        trace!("Drawn probability: {}", proba);
        if proba < acceptance_ratio {
            // Accept the update
            let worldlines_mut = system.path_mut();
            if let Some(modifications) = proposal.get_modifications(head) {
                for (range, new_positions) in modifications {
                    worldlines_mut.set_positions(head, range.start, range.end, new_positions);
                }
            }
            debug!("Move accepted");

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

// Inject the parameter tuning methods
impl_tunable_parameters!(RedrawHead, (min_delta_t, usize), (max_delta_t, usize));
