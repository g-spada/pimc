pub trait MonteCarloUpdate<W> {
    /// Apply a Monte Carlo update to the given worldlines.
    ///
    /// # Arguments
    /// - `worldlines`: The worldlines to update (generic type `W`).
    /// - `weight_function`: A closure that evaluates the weight of a configuration.
    /// - `rng`: A random number generator.
    ///
    /// # Returns
    /// `true` if the update was accepted, `false` otherwise.
    fn try_update(&mut self, worldlines: &mut W, rng: &mut impl rand::Rng) -> bool;
}
