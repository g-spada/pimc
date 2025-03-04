use super::traits::SystemAccess;
use crate::path::traits::{WorldLineDimensions, WorldLinePositionAccess};
use crate::space::traits::Space;

#[derive(Debug)]
pub struct HomonuclearSystem<S, W>
where
    S: Space,
    W: WorldLineDimensions + WorldLinePositionAccess,
{
    pub space: S,
    pub path: W,
    pub two_lambda_tau: f64,
}

impl<S, W> HomonuclearSystem<S, W>
where
    S: Space,
    W: WorldLineDimensions + WorldLinePositionAccess,
{
    pub fn reseat_polymer(&mut self, particle: usize) {
        // Validate input
        debug_assert!(particle < self.path.particles(), "Invalid particle index");
        debug_assert_eq!(
            S::SPATIAL_DIMENSIONS,
            W::SPATIAL_DIMENSIONS,
            "Spatial dimensions don't match"
        );

        // Compute the fundamental image of the first bead
        let first_image = self.space.base_image(self.path.position(particle, 0));

        // Compute the shift
        let shift = &first_image - &self.path.position(particle, 0);

        let mut whole_polymer = self.path.positions_mut(particle, 0, W::TIME_SLICES);

        // Add the shift to each bead of the polymer
        whole_polymer += &shift
            .view()
            .broadcast([W::TIME_SLICES, W::SPATIAL_DIMENSIONS])
            .unwrap();
    }
}

impl<S, W> SystemAccess for HomonuclearSystem<S, W>
where
    S: Space,
    W: WorldLineDimensions + WorldLinePositionAccess,
{
    type Space = S;
    type WorldLine = W;

    fn space(&self) -> &Self::Space {
        &self.space
    }

    fn path(&self) -> &Self::WorldLine {
        &self.path
    }

    fn path_mut(&mut self) -> &mut Self::WorldLine {
        &mut self.path
    }

    fn two_lambda_tau(&self, _: usize) -> f64 {
        self.two_lambda_tau
    }

    fn post_update_refactor(&mut self, particle: usize) {
        self.reseat_polymer(particle);
    }
}
