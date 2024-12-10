use super::accepted_update::AcceptedUpdate;

pub trait MonteCarloUpdate<W> {
    /// Apply a Monte Carlo update to the given worldlines.
    ///
    /// # Arguments
    /// - `worldlines`: The worldlines to update (generic type `W`).
    /// - `weight_function`: A closure that evaluates the weight of a configuration.
    /// - `rng`: A random number generator.
    ///
    /// # Returns
    /// An `Option<AcceptedUpdate>`.
    fn try_update(
        &mut self,
        worldlines: &mut W,
        rng: &mut impl rand::Rng,
    ) -> Option<AcceptedUpdate>;
}
