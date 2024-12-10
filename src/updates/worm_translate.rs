use super::monte_carlo_update::MonteCarloUpdate;
use super::proposed_update::ProposedUpdate;
use crate::path_state::sector::Sector;
use crate::path_state::traits::{
    WorldLineDimensions, WorldLinePermutationAccess, WorldLinePositionAccess, WorldLineWormAccess,
};
use crate::path_state::traverse_polymer::traverse_polymer;
use log::{debug, trace};
use ndarray::Array1;

/// A Monte Carlo update that translates both open and closed polymers.
pub struct WormTranslate<F, W, const D: usize> {
    max_displacement: [f64; D],
    weight_function: F,
    accept_count: usize,
    reject_count: usize,
    _phantom: std::marker::PhantomData<W>, // Ensures the type W is associated with the struct without requiring an actual field of type W
}

impl<F, W, const D: usize> WormTranslate<F, W, D>
where
    F: Fn(&W, &ProposedUpdate<f64>) -> f64 + Send + Sync + 'static,
    W: WorldLineDimensions
        + WorldLinePositionAccess
        + WorldLinePermutationAccess
        + WorldLineWormAccess,
{
    /// Initializes `WormTranslate` with the specified maximum displacements.
    ///
    /// # Arguments
    /// * `max_displacement` - An array specifying the maximum displacement for each dimension.
    pub fn new(max_displacement: [f64; D], weight_function: F) -> Self {
        Self {
            max_displacement,
            weight_function,
            accept_count: 0,
            reject_count: 0,
            _phantom: std::marker::PhantomData,
        }
    }

    fn select_initial_particle(&self, worldlines: &W, rng: &mut impl rand::Rng) -> usize {
        let mut p0: usize = rng.gen_range(0..worldlines.particles());
        if worldlines.sector() == Sector::G {
            // Traverse polymer to detect closed or open configuration.
            p0 = traverse_polymer(worldlines, p0);
        }
        p0
    }
}

impl<F, W, const D: usize> MonteCarloUpdate<W> for WormTranslate<F, W, D>
where
    F: Fn(&W, &ProposedUpdate<f64>) -> f64 + Send + Sync + 'static,
    W: WorldLineDimensions
        + WorldLinePositionAccess
        + WorldLinePermutationAccess
        + WorldLineWormAccess,
{
    fn try_update(&mut self, worldlines: &mut W, rng: &mut impl rand::Rng) -> bool {
        debug!("Trying update");
        let mut proposal = ProposedUpdate::new();

        //let tot_particles = worldlines.particles();
        let tot_slices = worldlines.time_slices();
        //let tot_dimensions = worldlines.spatial_dimensions();

        // Randomly select an initial particle index
        let p0 = self.select_initial_particle(worldlines, rng);
        // Generate displacement
        let displacement = Array1::from_shape_fn(self.max_displacement.len(), |i| {
            rng.gen_range(-self.max_displacement[i]..=self.max_displacement[i])
        });
        trace!("Displacement vector {:}", displacement);

        // Apply the displacement to the whole polymer
        let mut p = p0;
        loop {
            let new_positions = displacement.broadcast([tot_slices, D]).unwrap().to_owned()
                + worldlines.positions(p0, 0, tot_slices);
            debug_assert_eq!(
                new_positions.shape()[0],
                tot_slices,
                "Expected positions to match time slices"
            );

            proposal.add_position_modification(p, 0..tot_slices, new_positions);
            if let Some(next) = worldlines.following(p) {
                if next == p0 {
                    // End of the cycle.
                    break;
                }
                p = next;
            } else {
                // Polymer is worm: Head reached.
                break;
            }
        }

        let acceptance_ratio = (self.weight_function)(worldlines, &proposal);
        trace!("Acceptance ratio {:}", acceptance_ratio);

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
            true
        } else {
            // Reject the update
            self.reject_count += 1;
            debug!("Move rejected");
            false
        }
    }
}
