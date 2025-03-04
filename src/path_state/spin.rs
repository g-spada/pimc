use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Spin {
    Up,
    Dn,
}

impl Spin {
    pub fn flip(&mut self) {
        *self = match self {
            Spin::Up => Spin::Dn,
            Spin::Dn => Spin::Up,
        };
    }
}
