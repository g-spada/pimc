use crate::path_state::traits::WorldLineDimensions;
use crate::space::traits::Space;

pub trait SystemAccess {
    type Space: Space;
    type WorldLine: WorldLineDimensions;

    fn space(&self) -> &Self::Space;
    fn two_lambda_tau(&self, particle: usize) -> f64;
    fn path(&self) -> &Self::WorldLine;
    fn path_mut(&mut self) -> &mut Self::WorldLine;
}
