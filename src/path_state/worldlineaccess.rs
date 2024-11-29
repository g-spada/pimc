use ndarray::{Array2, ArrayView1, ArrayView2, ArrayViewMut1};

/// A trait to abstract access and manipulation of particle worldlines.
///
/// The `WorldLineAccess` trait provides a generic interface for working with particle worldlines
/// in Monte Carlo simulations. Implementing this trait allows for easy manipulation of particle
/// positions, states, and their time evolution without depending on a specific worldline structure.
///
/// # Associated Types
/// - `State`: Represents the quantum state of a particle, which can be customized.
pub trait WorldLineAccess {
    /// Associated type for the particle quantum state type.
    type State;

    /// Returns the number of particles in the system.
    ///
    /// # Returns
    /// The total number of particles.
    fn particles(&self) -> usize;

    /// Returns the number of time slices in the system.
    ///
    /// # Returns
    /// The total number of time slices.
    fn time_slices(&self) -> usize;

    /// Returns the number of spatial dimensions.
    ///
    /// # Returns
    /// The spatial dimensionality of the system.
    fn spatial_dimentions(&self) -> usize;

    /// Gets a view of the position of a specific particle at a specific time slice.
    ///
    /// # Arguments
    /// - `particle`: The index of the particle.
    /// - `time_slice`: The index of the time slice.
    ///
    /// # Returns
    /// An immutable 1D array view of the particle's position.
    fn position(&self, particle: usize, time_slice: usize) -> ArrayView1<f64>;

    /// Gets a mutable view of the position of a specific particle at a specific time slice.
    ///
    /// # Arguments
    /// - `particle`: The index of the particle.
    /// - `time_slice`: The index of the time slice.
    ///
    /// # Returns
    /// A mutable 1D array view of the particle's position.
    fn position_mut(&mut self, particle: usize, time_slice: usize) -> ArrayViewMut1<f64>;

    /// Sets the position of a specific particle at a specific time slice.
    ///
    /// # Arguments
    /// - `particle`: The index of the particle.
    /// - `time_slice`: The index of the time slice.
    /// - `bead_position`: A slice containing the new position values.
    fn set_position(&mut self, particle: usize, time_slice: usize, bead_position: &[f64]);

    /// Gets a view of the positions for a specific particle across a range of time slices.
    ///
    /// # Arguments
    /// - `particle`: The index of the particle.
    /// - `start_slice`: The starting index of the time slice (inclusive).
    /// - `end_slice`: The ending index of the time slice (exclusive).
    ///
    /// # Returns
    /// A 2D array view of the particle's positions for the specified time range.
    fn positions(&self, particle: usize, start_slice: usize, end_slice: usize) -> ArrayView2<f64>;

    /// Sets the positions for a specific particle across a range of time slices.
    ///
    /// # Arguments
    /// - `particle`: The index of the particle.
    /// - `start_slice`: The starting index of the time slice (inclusive).
    /// - `end_slice`: The ending index of the time slice (exclusive).
    /// - `positions`: A 2D array containing the new positions.
    fn set_positions(
        &mut self,
        particle: usize,
        start_slice: usize,
        end_slice: usize,
        positions: &Array2<f64>,
    );

    /// Gets the state of a specific particle at a specific time slice.
    ///
    /// # Arguments
    /// - `particle`: The index of the particle.
    /// - `time_slice`: The index of the time slice.
    ///
    /// # Returns
    /// The state of the particle at the given time slice.
    fn state(&self, particle: usize, time_slice: usize) -> Self::State;

    /// Sets the state of a specific particle at a specific time slice.
    ///
    /// # Arguments
    /// - `particle`: The index of the particle.
    /// - `time_slice`: The index of the time slice.
    /// - `state`: The new state to assign to the particle.
    fn set_state(&mut self, particle: usize, time_slice: usize, state: Self::State);
}
