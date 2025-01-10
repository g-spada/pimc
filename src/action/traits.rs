use crate::system::traits::SystemAccess;
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
pub trait PotentialDensityMatrix<S: SystemAccess> {
    fn potential_density_matrix(&self, system: &S) -> f64;
    fn potential_density_matrix_position_update(
        &self,
        system: &S,
        update: &ProposedUpdate<f64>,
    ) -> f64;
}
