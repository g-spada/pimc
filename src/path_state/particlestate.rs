use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ParticleState {
    Up,
    Down,
}

//impl Default for ParticleState {
    //fn default() -> Self {
        //ParticleState::Up
    //}
//}
