//! Macros for implementing Monte Carlo simulation components.
//!
//! Provides declarative macros to reduce boilerplate in:
//! - Parameter tuning (`impl_tunable_parameters!`),
//! - Update enum generation (`define_update_enum!`),
//! - Simulation container setup (`define_pimc!`).

/// Implements the `MonteCarloTunable` trait for a type with named parameters.
///
/// # Usage
/// ```ignore
/// impl_tunable_parameters!(MyUpdate, (temperature, f64), (steps, usize));
/// ```
///
/// Expands to `set_parameter`/`get_parameter` implementations that:
/// - Convert `f64` values to the specified type (`$type`).
/// - Return `bool`/`Option<f64>` to signal success.
#[macro_export]
macro_rules! impl_tunable_parameters {
    ($update:ident, $(($param:ident, $type:ty)),+) => {
        impl MonteCarloTunable for $update {
            fn set_parameter(&mut self, param_name: &str, value: f64) -> bool {
                match param_name {
                    $(
                        // Match parameter name and cast value to the target type
                        stringify!($param) => {
                            self.$param = value as $type;
                            true
                        }
                    )+
                    _ => false, // Unknown parameter
                }
            }

            fn get_parameter(&self, param_name: &str) -> Option<f64> {
                match param_name {
                    $(stringify!($param) => Some(self.$param as f64),)+
                    _ => None,
                }
            }
        }
    };
}

/// Defines an enum to represent multiple update types with automatic trait delegation.
///
/// # Usage
/// ```ignore
/// define_update_enum!(MyUpdatesEnum, MyUpdate1, MyUpdate2);
/// ```
///
/// Generates:
/// - An enum with variants for each update type.
/// - `MonteCarloStep`, `MonteCarloTunable`, and `From<T>` implementations.
/// - A `type_name()` method for runtime type inspection.
#[macro_export]
macro_rules! define_update_enum {
    ($name:ident, $($update:ident),+) => {
        /// Enum wrapping all possible update types.
        pub enum $name {
            $(
                $update($update), // Variant per update type
            )+
        }

        impl $name {
            /// Returns the concrete type name of the update variant.
            pub fn type_name(&self) -> &'static str {
                match self {
                    $(
                        Self::$update(_) => stringify!($update),
                    )+
                }
            }
        }

        // Delegate MonteCarloStep to the inner update type
        impl<S, A, R> $crate::monte_carlo::traits::MonteCarloStep<S, A, R> for $name
        where
            R: rand::Rng,
            $(
                $update: $crate::monte_carlo::traits::MonteCarloStep<S, A, R>,
            )+
        {
            fn step(
                &mut self,
                system: &mut S,
                action: &A,
                rng: &mut R,
            ) -> Option<$crate::monte_carlo::accepted_update::AcceptedUpdate> {
                match self {
                    $(
                        Self::$update(inner) =>
                            <$update as $crate::monte_carlo::traits::MonteCarloStep<S, A, R>>::step(
                                inner, system, action, rng
                            ),
                    )+
                }
            }
        }

        // Delegate MonteCarloTunable to the inner update type
        impl $crate::monte_carlo::traits::MonteCarloTunable for $name {
            fn set_parameter(&mut self, param_name: &str, value: f64) -> bool {
                match self {
                    $(
                        Self::$update(inner) =>
                            <$update as $crate::monte_carlo::traits::MonteCarloTunable>::set_parameter(
                                inner, param_name, value
                            ),
                    )+
                }
            }

            fn get_parameter(&self, param_name: &str) -> Option<f64> {
                match self {
                    $(
                        Self::$update(inner) =>
                            <$update as $crate::monte_carlo::traits::MonteCarloTunable>::get_parameter(
                                inner, param_name
                            ),
                    )+
                }
            }
        }

        // Auto-generate From<T> impls for each variant
        $(
            impl From<$update> for $name {
                fn from(update: $update) -> Self {
                    Self::$update(update)
                }
            }
        )+
    };
}

/// Defines a complete Monte Carlo simulation container with updates and statistics.
///
/// # Usage
/// ```
/// use pimc::define_pimc;
/// use pimc::updates::{Redraw, Translate};
/// define_pimc!(MyPimc, [Redraw, Translate]);
/// ```
///
/// # Implementation Details
/// The macro generates:
/// 1. **An update enum** (via `define_update_enum!`) to hold heterogeneous updates.
/// 2. **A simulation struct** containing:
///    - System state (`S`), action (`A`), and RNG (`R`).
///    - A list of updates and their statistics.
/// 3. **Trait implementations**:
///    - `MonteCarloStep` for sweeping/stepping updates.
///    - `UpdateType<T>` for type-safe parameter access.
/// 4. **Statistics tracking** with formatted output.
///
/// # Notes
/// - Panics if duplicate update types are added.
/// - Uses `paste::paste!` for hygienic identifier generation.
#[macro_export]
macro_rules! define_pimc {
    ($name:ident, [ $($update_type:ident),+ ]) => {
        paste::paste! {
            // Generate the update enum (e.g., `MyUpdatesEnum`)
            $crate::define_update_enum!([<$name Updates>], $($update_type),+);

            /// Monte Carlo simulation container.
            pub struct $name<S, A, R> {
                pub system: S,
                pub action: A,
                rng: R,
                updates: Vec<[<$name Updates>]>,
                stats: Vec<$crate::monte_carlo::update_stats::UpdateStats>,
            }

            impl<S, A, R: rand::Rng> $name<S, A, R> {
                /// Creates a new simulation with empty updates and stats.
                pub fn new(system: S, action: A, rng: R) -> Self {
                    Self {
                        system,
                        action,
                        rng,
                        updates: Vec::new(),
                        stats: Vec::new(),
                    }
                }

                /// Adds an update, ensuring no duplicates.
                pub fn with_update<T>(mut self, update: T) -> Self
                where
                    T: Into<[<$name Updates>]>,
                    [<$name Updates>]: $crate::monte_carlo::traits::UpdateType<T>,
                {
                    let update = update.into();
                    // Check for existing update of the same type
                    if self.updates.iter().any(|u| {
                        <[<$name Updates>] as $crate::monte_carlo::traits::UpdateType<T>>::is_type(u)
                    }) {
                        panic!(
                            "Update type {} already exists in {}",
                            update.type_name(),
                            stringify!($name)
                        );
                    }
                    self.updates.push(update);
                    self.stats.push($crate::monte_carlo::update_stats::UpdateStats::new());
                    self
                }

                /// Executes all updates in sequence (a "sweep").
                pub fn sweep(&mut self)
                where
                    [<$name Updates>]: $crate::monte_carlo::traits::MonteCarloStep<S, A, R>,
                {
                    for (update, stats) in self.updates.iter_mut().zip(self.stats.iter_mut()) {
                        stats.attempts += 1;
                        if <[<$name Updates>] as $crate::monte_carlo::traits::MonteCarloStep<S, A, R>>::step(
                            update, &mut self.system, &self.action, &mut self.rng
                        ).is_some() {
                            stats.accepted += 1;
                        }
                    }
                }

                /// Executes a single step of update type `T`.
                pub fn step<T>(&mut self) -> Option<$crate::monte_carlo::accepted_update::AcceptedUpdate>
                where
                    [<$name Updates>]: $crate::monte_carlo::traits::UpdateType<T>,
                    [<$name Updates>]: $crate::monte_carlo::traits::MonteCarloStep<S, A, R>,
                {
                    for (update, stats) in self.updates.iter_mut().zip(self.stats.iter_mut()) {
                        if <[<$name Updates>] as $crate::monte_carlo::traits::UpdateType<T>>::is_type(update) {
                            stats.attempts += 1;
                            let result = <[<$name Updates>] as $crate::monte_carlo::traits::MonteCarloStep<S, A, R>>::step(
                                update, &mut self.system, &self.action, &mut self.rng
                            );
                            if result.is_some() {
                                stats.accepted += 1;
                            }
                            return result;
                        }
                    }
                    None
                }

                /// Sets a parameter for the update of type `T`.
                pub fn set_update_parameter<T>(&mut self, param_name: &str, value: f64)
                where
                    T: $crate::monte_carlo::traits::MonteCarloTunable,
                    [<$name Updates>]: $crate::monte_carlo::traits::UpdateType<T>,
                {
                    for update in &mut self.updates {
                        if <[<$name Updates>] as $crate::monte_carlo::traits::UpdateType<T>>::set_param_if_match(
                            update, param_name, value
                        ) {
                            break;
                        }
                    }
                }

                /// Get a parameter from the update of type `T`.
                pub fn get_update_parameter<T>(&self, param_name: &str) -> Option<f64>
                where
                    T: $crate::monte_carlo::traits::MonteCarloTunable,
                    [<$name Updates>]: $crate::monte_carlo::traits::UpdateType<T>,
                {
                    for update in &self.updates {
                        if let Some(value) = <[<$name Updates>] as $crate::monte_carlo::traits::UpdateType<T>>::get_param_if_match(
                            update, param_name
                        ) {
                            return Some(value);
                        }
                    }
                    None
                }

                /// Returns formatted statistics (acceptance rates, etc.).
                pub fn stats(&self) -> String {
                    use std::fmt::Write;

                    // Calculate maximum widths for each column
                    let mut max_update_width = "UPDATE".len() + 2;
                    let mut max_accepted_width = "ACCEPTED".len() + 2;
                    let mut max_attempts_width = "ATTEMPTS".len() + 2;

                    for (update, stats) in self.updates.iter().zip(self.stats.iter()) {
                        max_update_width = max_update_width.max(update.type_name().len() + 2 );
                        max_accepted_width = max_accepted_width.max(stats.accepted.to_string().len() + 2);
                        max_attempts_width = max_attempts_width.max(stats.attempts.to_string().len() + 2);
                    }

                    let mut output = String::new();
                    // Header
                    writeln!(output, "Statistics for {}:", stringify!($name)).unwrap();
                    writeln!(
                        output,
                        "  {update:<update_width$} {accepted:>accepted_width$} {attempts:>attempts_width$} {rate:>6}",
                        update = "UPDATE",
                        accepted = "ACCEPTED",
                        attempts = "ATTEMPTS",
                        rate = "RATE",
                        update_width = max_update_width,
                        accepted_width = max_accepted_width,
                        attempts_width = max_attempts_width
                    ).unwrap();

                    // Data rows
                    for (update, stats) in self.updates.iter().zip(self.stats.iter()) {
                        let acceptance_rate = if stats.attempts > 0 {
                            stats.accepted as f64 / stats.attempts as f64
                        } else {
                            0.0
                        };
                        writeln!(
                            output,
                            "  {update:<update_width$} {accepted:>accepted_width$} {attempts:>attempts_width$} {rate:>6.2}",
                            update = update.type_name(),
                            accepted = stats.accepted,
                            attempts = stats.attempts,
                            rate = acceptance_rate,
                            update_width = max_update_width,
                            accepted_width = max_accepted_width,
                            attempts_width = max_attempts_width
                        ).unwrap();
                    }
                    output
                }

                pub fn reset_stats(&mut self) {
                    for stats in &mut self.stats {
                        stats.reset();
                    }
                }

                pub fn reset_stats_for<T>(&mut self)
                where
                    [<$name Updates>]: $crate::monte_carlo::traits::UpdateType<T>,
                {
                    for (update, stats) in self.updates.iter().zip(self.stats.iter_mut()) {
                        if <[<$name Updates>] as $crate::monte_carlo::traits::UpdateType<T>>::is_type(update) {
                            stats.reset();
                            break;
                        }
                    }
                }
            }

            // Implement UpdateType for each variant of the enum
            $(
                impl $crate::monte_carlo::traits::UpdateType<$update_type> for [<$name Updates>] {
                    fn set_param_if_match(&mut self, param_name: &str, value: f64) -> bool {
                        if let Self::$update_type(inner) = self {
                            <$update_type as $crate::monte_carlo::traits::MonteCarloTunable>::set_parameter(
                                inner, param_name, value
                            )
                        } else {
                            false
                        }
                    }

                    fn get_param_if_match(&self, param_name: &str) -> Option<f64> {
                        if let Self::$update_type(inner) = self {
                            <$update_type as $crate::monte_carlo::traits::MonteCarloTunable>::get_parameter(
                                inner, param_name
                            )
                        } else {
                            None
                        }
                    }

                    fn is_type(&self) -> bool {
                        matches!(self, Self::$update_type(_))
                    }
                }
            )+
        }
    };
}
