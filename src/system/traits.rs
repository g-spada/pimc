use crate::space::traits::Space;
use crate::path_state::traits::WorldLineDimensions;

pub trait SystemAccess {
    type Space: Space;
    type WorldLine: WorldLineDimensions;

    fn space(&self) -> &Self::Space;
    fn path(&self) -> &Self::WorldLine;
    fn path_mut(&mut self) -> &mut Self::WorldLine;
}
