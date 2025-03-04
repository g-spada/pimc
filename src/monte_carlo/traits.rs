use crate::monte_carlo::accepted_update::AcceptedUpdate;

/// A trait defining the behavior of a Monte Carlo simulation step.
///
/// Implementors of this trait must define how a single Monte Carlo step is applied,
/// given a system state, an action, and a random number generator.
///
/// # Type Parameters
/// - `S`: The type representing the system state being modified.
/// - `A`: The type representing the action taken in the step.
/// - `R`: The random number generator type (must implement `rand::Rng`).
///
/// # Returns
/// - `Option<AcceptedUpdate>`:
///   - `Some(update)` if the step was accepted (contains metadata about the update).
///   - `None` if the step was rejected.
pub trait MonteCarloStep<S, A, R: rand::Rng> {
    /// Performs a single Monte Carlo step.
    ///
    /// # Arguments
    /// - `system`: Mutable reference to the system state.
    /// - `action`: Reference to the action applied.
    /// - `rng`: Mutable reference to the random number generator.
    fn step(&mut self, system: &mut S, action: &A, rng: &mut R) -> Option<AcceptedUpdate>;
}

/// A trait for tuning numerical parameters in a Monte Carlo simulation.
///
/// Provides methods to dynamically get/set parameters by name (string-based lookup).
/// Default implementations return `false`/`None` for unimplemented parameters.
pub trait MonteCarloTunable {
    /// Sets a parameter's value by name.
    ///
    /// # Arguments
    /// - `param_name`: Name of the parameter to set.
    /// - `value`: New value (converted to the parameter's type internally).
    ///
    /// # Returns
    /// - `true` if the parameter was found and updated.
    /// - `false` if the parameter doesn't exist or conversion failed.
    fn set_parameter(&mut self, _param_name: &str, _value: f64) -> bool {
        false
    }

    /// Gets a parameter's value by name.
    ///
    /// # Arguments
    /// - `param_name`: Name of the parameter to query.
    ///
    /// # Returns
    /// - `Some(value)` if the parameter exists (converted to `f64`).
    /// - `None` if the parameter doesn't exist.
    fn get_parameter(&self, _param_name: &str) -> Option<f64> {
        None
    }
}

/// A helper trait for type-parameterized parameter access.
///
/// Used to check and manipulate parameters only if the implementing type matches `T`.
pub trait UpdateType<T> {
    /// Attempts to set a parameter if the implementing type matches `T`.
    ///
    /// # Arguments
    /// - `param_name`: Name of the parameter to set.
    /// - `value`: New value (as `f64`).
    ///
    /// # Returns
    /// - `true` if the type matched and the parameter was set.
    /// - `false` otherwise.
    fn set_param_if_match(&mut self, param_name: &str, value: f64) -> bool;

    /// Attempts to get a parameter if the implementing type matches `T`.
    ///
    /// # Arguments
    /// - `param_name`: Name of the parameter to query.
    ///
    /// # Returns
    /// - `Some(value)` if the type matched and the parameter exists.
    /// - `None` otherwise.
    fn get_param_if_match(&self, param_name: &str) -> Option<f64>;

    /// Checks if the implementing type matches `T`.
    ///
    /// # Returns
    /// - `true` if the type matches.
    /// - `false` otherwise.
    fn is_type(&self) -> bool;
}
