use super::accepted_update::AcceptedUpdate;
use super::monte_carlo_update::MonteCarloUpdate;
use super::proposed_update::ProposedUpdate;
use crate::action::traits::PotentialDensityMatrix;
use crate::path_state::sector::Sector;
use crate::path_state::traits::{
    WorldLineDimensions, WorldLinePermutationAccess, WorldLinePositionAccess, WorldLineWormAccess,
};
use crate::path_state::traverse_polymer::traverse_polymer;
use crate::system::traits::SystemAccess;
use log::{debug, trace};
use ndarray::Array1;

/// A Monte Carlo update that translates both open and closed polymers.
pub struct WormTranslate {
    pub max_displacement: f64,
    pub accept_count: usize,
    pub reject_count: usize,
}

impl WormTranslate {
    fn select_initial_particle<W>(&self, worldlines: &W, rng: &mut impl rand::Rng) -> usize
    where
        W: WorldLinePermutationAccess + WorldLineWormAccess + WorldLineDimensions,
    {
        let mut p0: usize = rng.gen_range(0..worldlines.particles());
        if worldlines.sector() == Sector::G {
            // Traverse polymer to detect closed or open configuration.
            p0 = traverse_polymer(worldlines, p0);
        }
        p0
    }
}

impl<S, A> MonteCarloUpdate<S, A> for WormTranslate
where
    S: SystemAccess,
    S::WorldLine: WorldLineDimensions
        + WorldLinePositionAccess
        + WorldLinePermutationAccess
        + WorldLineWormAccess,
    A: PotentialDensityMatrix,
{
    fn monte_carlo_update(
        &mut self,
        system: &mut S,
        action: &A,
        rng: &mut impl rand::Rng,
    ) -> Option<AcceptedUpdate> {
        debug!("Trying update");
        let mut proposal = ProposedUpdate::new();

        let worldlines = system.path();
        //let tot_particles = worldlines.particles();
        let tot_slices = S::WorldLine::TIME_SLICES;
        let tot_directions = S::WorldLine::SPATIAL_DIMENSIONS;
        // Find head of the worm
        // Randomly select an initial particle index
        let p0 = self.select_initial_particle(worldlines, rng);

        // Generate displacement
        let displacement: Array1<f64> = (0..tot_directions)
            .map(|_| rng.gen_range(-self.max_displacement..=self.max_displacement))
            .collect();
        trace!("Displacement vector {:?}", displacement);

        // Apply the displacement to the whole polymer
        let mut p = p0;
        loop {
            let new_positions = displacement
                .broadcast([tot_slices, tot_directions])
                .unwrap()
                .to_owned()
                + worldlines.positions(p, 0, tot_slices);
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

        let acceptance_ratio = action.potential_density_matrix_update(system, &proposal);
        trace!("Acceptance ratio {:}", acceptance_ratio);

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
