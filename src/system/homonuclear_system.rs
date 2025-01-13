use super::traits::SystemAccess;
use crate::path_state::traits::{WorldLineDimensions, WorldLinePositionAccess};
use crate::space::traits::Space;

pub struct HomonuclearSystem<S, W> 
where
    S : Space,
    W: WorldLineDimensions + WorldLinePositionAccess,
{
    pub space: S,
    pub path: W,
    pub two_lambda_tau: f64,
}

impl<S, W> SystemAccess for HomonuclearSystem<S, W>
where
    S : Space,
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
}

