use super::accepted_update::AcceptedUpdate;

pub trait MonteCarloUpdate<S,A> {
    /// Apply the Monte Carlo update.
    ///
    /// # Returns
    /// An `Option<AcceptedUpdate>`.
    fn monte_carlo_update(
        &mut self,
        system: &mut S,
        action: &A,
        rng: &mut impl rand::Rng,
    ) -> Option<AcceptedUpdate>;
}
