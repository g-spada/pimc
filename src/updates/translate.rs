use super::monte_carlo_update::MonteCarloUpdate;
use super::proposed_update::ProposedUpdate;
use crate::path_state::sector::Sector;
use crate::path_state::traits::WorldLineWithPermutations;
use ndarray::{Array1, Array2, Axis};

/// A simple mock Monte Carlo update for testing.
pub struct Translate {
    max_displacement: Vec<f64>,
}

impl Translate {
    /// Initializes `Translate` with the specified maximum displacements.
    ///
    /// # Arguments
    /// * `max_displacement` - A vector specifying the maximum displacement for each dimension.
    pub fn new(max_displacement: Vec<f64>) -> Self {
        Self { max_displacement }
    }
}

impl<W> MonteCarloUpdate<W, f64> for Translate
where
    W: WorldLineWithPermutations, // Ensure W implements WorldLineAccess with the given State type
{
    fn try_update<F>(
        &self,
        worldlines: &mut W,
        weight_function: F,
        rng: &mut impl rand::Rng,
    ) -> bool
    where
        F: Fn(&W, &ProposedUpdate<f64>) -> f64,
    {
        let mut proposal = ProposedUpdate::new();
        // Randomly select an initial particle index
        let mut p0: usize = rng.gen_range(0..worldlines.particles());
        if worldlines.sector() == Sector::G {
            // Navigate the polymer to detect if current polymer is closed.
            // We keep p0 if polymer is closed, or we set `p0 = tail` if polymer is open.
            let mut p = p0;
            p0 = loop {
                if let Some(next) = worldlines.preceding(p) {
                    if next == p0 {
                        break p0; // Cycle detected, return the start value
                    }
                    p = next;
                } else {
                    break p; // End reached, return the last valid value
                }
            };
        }
        let d = worldlines.spatial_dimensions();
        let t = worldlines.time_slices();
        // Generate displacement
        let displacements = Array1::from_iter((0..d).map(|_| {
            rng.gen_range(-1.0..=1.0) * self.max_displacement[d % self.max_displacement.len()]
        }));

        // Apply the displacement to the whole polymer
        let mut p = p0;
        loop {
            let mut new_positions: Array2<f64> = worldlines.positions(p0, 0, t).to_owned();
            new_positions
                .axis_iter_mut(Axis(0))
                .for_each(|mut slice| slice += &displacements);
            proposal.add_position_modification(p0, 0..t, new_positions);
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

        let acceptance_ratio = weight_function(&worldlines, &proposal);

        // Apply Metropolis-Hastings acceptance criterion
        if rng.gen::<f64>() < acceptance_ratio {
            // Accept the update
            for particle in proposal.get_modified_particles() {
                if let Some(modifications) = proposal.get_modifications(particle) {
                    for (range, new_positions) in modifications {
                        worldlines.set_positions(particle, range.start, range.end, &new_positions);
                    }
                }
            }
            true
        } else {
            // Reject the update
            false
        }
    }
}
