use super::sector::Sector;
use ndarray::{Array2, ArrayView1, ArrayView2, ArrayViewMut1, ArrayViewMut2};

/// Trait for querying the dimensional properties of worldlines.
pub trait WorldLineDimensions {
    /// Returns the number of particles in the system.
    fn particles(&self) -> usize;

    /// Returns the number of time slices in the system.
    fn time_slices(&self) -> usize;

    /// Returns the number of spatial dimensions.
    fn spatial_dimensions(&self) -> usize;
}

/// Trait for accessing and modifying particle positions in worldlines.
pub trait WorldLinePositionAccess {
    /// Gets a view of the position of a specific particle at a specific time slice.
    fn position(&self, particle: usize, time_slice: usize) -> ArrayView1<f64>;

    /// Gets a mutable view of the position of a specific particle at a specific time slice.
    fn position_mut(&mut self, particle: usize, time_slice: usize) -> ArrayViewMut1<f64>;

    /// Sets the position of a specific particle at a specific time slice.
    fn set_position(&mut self, particle: usize, time_slice: usize, bead_position: &[f64]);

    /// Gets a view of the positions for a specific particle across a range of time slices.
    fn positions(&self, particle: usize, start_slice: usize, end_slice: usize) -> ArrayView2<f64>;

    /// Gets a view of the positions for a specific particle across a range of time slices.
    fn positions_mut(&mut self, particle: usize, start_slice: usize, end_slice: usize) -> ArrayViewMut2<f64>;

    /// Sets the positions for a specific particle across a range of time slices.
    fn set_positions(
        &mut self,
        particle: usize,
        start_slice: usize,
        end_slice: usize,
        positions: &Array2<f64>,
    );
}

/// Trait for accessing the permutation structure of worldlines.
pub trait WorldLinePermutationAccess {
    /// Gets the index of the preceding particle in the polymer.
    fn preceding(&self, particle: usize) -> Option<usize>;

    ///// Sets the index of the preceding particle in the polymer.
    //fn set_preceding(&mut self, particle: usize, preceding: Option<usize>);

    /// Gets the index of the following particle in the polymer.
    fn following(&self, particle: usize) -> Option<usize>;

    ///// Sets the index of the following particle in the polymer.
    //fn set_following(&mut self, particle: usize, following: Option<usize>);
}

/// Trait for accessing the permutation structure of worldlines.
pub trait WorldLineWormAccess {
    /// Gets the index of the worm head, if it exists.
    fn worm_head(&self) -> Option<usize>;

    /// Gets the index of the worm tail, if it exists.
    fn worm_tail(&self) -> Option<usize>;

    /// Gets the sector of the worldlines.
    fn sector(&self) -> Sector;
}

/// Trait for accessing and modifying particle internal quantum states of the worldlines.
pub trait WorldLineStateAccess {
    /// Associated type for the particle quantum state type.
    type State;

    /// Gets the state of a specific particle at a specific time slice.
    fn state(&self, particle: usize, time_slice: usize) -> Self::State;

    /// Sets the state of a specific particle at a specific time slice.
    fn set_state(&mut self, particle: usize, time_slice: usize, state: Self::State);
}

//pub trait WorldLineBase: WorldLineDimensions + WorldLinePositionAccess {}
//pub trait WorldLineWithPermutations: WorldLineBase + WorldLinePermutationAccess {}
