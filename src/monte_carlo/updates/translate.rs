use crate::action::traits::PotentialDensityMatrix;
use crate::impl_tunable_parameters;
use crate::monte_carlo::accepted_update::AcceptedUpdate;
use crate::monte_carlo::proposed_update::ProposedUpdate;
use crate::monte_carlo::traits::{MonteCarloStep, MonteCarloTunable};
use crate::path::sector::Sector;
use crate::path::traits::{
    WorldLineDimensions, WorldLinePermutationAccess, WorldLinePositionAccess, WorldLineWormAccess,
};
use crate::path::traverse_polymer::traverse_polymer;
use crate::system::traits::SystemAccess;
use log::{debug, trace};
use ndarray::Array1;

/// A Monte Carlo update that translates both open and closed polymers.
#[derive(Debug, Clone, Copy)]
pub struct Translate {
    max_displacement: f64,
}

impl Translate {
    fn select_initial_particle<W>(&self, worldlines: &W, rng: &mut impl rand::Rng) -> usize
    where
        W: WorldLinePermutationAccess + WorldLineWormAccess + WorldLineDimensions,
    {
        let mut p0: usize = rng.random_range(0..worldlines.particles());
        if worldlines.sector() == Sector::G {
            // Traverse polymer to detect closed or open configuration.
            p0 = traverse_polymer(worldlines, p0);
        }
        p0
    }

    pub fn new(max_displacement: f64) -> Self {
        Self { max_displacement }
    }
}

impl<S, A, R> MonteCarloStep<S, A, R> for Translate
where
    S: SystemAccess,
    S::WorldLine: WorldLineDimensions
        + WorldLinePositionAccess
        + WorldLinePermutationAccess
        + WorldLineWormAccess,
    A: PotentialDensityMatrix,
    R: rand::Rng,
{
    fn step(&mut self, system: &mut S, action: &A, rng: &mut R) -> Option<AcceptedUpdate> {
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
            .map(|_| rng.random_range(-self.max_displacement..=self.max_displacement))
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

            // Bring the modified polymer to its standard form
            for particle in proposal.get_modified_particles() {
                system.post_update_refactor(particle);
            }

            Some(proposal.to_accepted_update())
        } else {
            // Reject the update
            debug!("Move rejected");
            None
        }
    }
}

// Inject the parameter tuning methods
impl_tunable_parameters!(Translate, (max_displacement, f64));
