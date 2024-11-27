use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ParticleState {
    Up,
    Dn,
}

//impl Default for ParticleState {
//fn default() -> Self {
//ParticleState::Up
//}
//}
