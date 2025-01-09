use crate::updates::proposed_update::ProposedUpdate;

/// One-Body external potential
pub trait OneBodyPotential {
    fn one_body_potential(&self, position: &[f64]) -> f64;
}

/// Two-Body potential interaction
pub trait PairPotential {
    fn pair_potential(&self, points_difference: &[f64]) -> f64;
}

/// Potential density matrix
pub trait PotentialDensityMatrix {
    fn potential_density_matrix(&self) -> f64;
    fn potential_density_matrix_update(&self, update: &ProposedUpdate<f64>) -> f64;
}
