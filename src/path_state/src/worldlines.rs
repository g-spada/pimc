use ndarray::{arr1, s, Array3, ArrayView1, ArrayViewMut1};

/// WorldLines struct to store particle positions using ndarray.
///
/// # Example
/// ```
/// use path_state::WorldLines;
///
/// // Create a new WorldLines instance
/// let mut world = WorldLines::new(2, 3, 3); // 2 particles, 3 time slices, 3D space
///
/// // Set the position of the first particle at the first time slice
/// world.set_position(0, 0, &[1.0, 2.0, 3.0]);
///
/// // Retrieve the position
/// let position = world.get_position(0, 0);
/// assert_eq!(position.to_vec(), vec![1.0, 2.0, 3.0]);
/// ```
///
pub struct WorldLines {
    /// Multidimensional array with dimensions (N particles, M time slices, D spatial dimensions).
    positions: Array3<f64>,
}

impl WorldLines {
    /// Creates a new `WorldLines` instance with all positions initialized to zero.
    ///
    /// # Arguments
    /// * `n` - Number of particles.
    /// * `m` - Number of time slices.
    /// * `d` - Spatial dimensionality.
    pub fn new(n: usize, m: usize, d: usize) -> Self {
        // Ensure no zero dimensions are passed (unphysical configuration)

        assert!(n > 0, "Number of particles (n) must be greater than 0.");
        assert!(m > 0, "Number of time slices (m) must be greater than 0.");
        assert!(d > 0, "Spatial dimensionality (d) must be greater than 0.");

        Self {
            positions: Array3::zeros((n, m, d)),
        }
    }

    /// Gets a view of the position of a specific particle at a specific time slice.
    ///
    /// # Arguments
    /// * `particle` - Index of the particle.
    /// * `time_slice` - Index of the time slice.
    ///
    /// # Returns
    /// An array view of the positions.
    ///
    /// # Panics
    /// Panics if the particle or time_slice indices are out of bounds.
    pub fn get_position(&self, particle: usize, time_slice: usize) -> ArrayView1<f64> {
        // Ensure indices are within bounds
        assert!(
            particle < self.positions.shape()[0],
            "Particle index out of bounds: particle={}, max allowed={}",
            particle,
            self.positions.shape()[0] - 1
        );
        assert!(
            time_slice < self.positions.shape()[1],
            "Time slice index out of bounds: time_slice={}, max allowed={}",
            time_slice,
            self.positions.shape()[1] - 1
        );

        self.positions.slice(s![particle, time_slice, ..])
    }

    /// Gets a mutable view of the position of a specific particle at a specific time slice.
    ///
    /// # Arguments
    /// * `particle` - Index of the particle.
    /// * `time_slice` - Index of the time slice.
    ///
    /// # Returns
    /// A mutable array view of the positions.
    ///
    /// # Panics
    /// Panics if the particle or time_slice indices are out of bounds.
    pub fn get_position_mut(&mut self, particle: usize, time_slice: usize) -> ArrayViewMut1<f64> {
        // Ensure indices are within bounds
        assert!(
            particle < self.positions.shape()[0],
            "Particle index out of bounds: particle={}, max allowed={}",
            particle,
            self.positions.shape()[0] - 1
        );
        assert!(
            time_slice < self.positions.shape()[1],
            "Time slice index out of bounds: time_slice={}, max allowed={}",
            time_slice,
            self.positions.shape()[1] - 1
        );

        self.positions.slice_mut(s![particle, time_slice, ..])
    }

    /// Sets the position of a specific particle at a specific time slice.
    ///
    /// # Arguments
    /// * `particle` - Index of the particle.
    /// * `time_slice` - Index of the time slice.
    /// * `bead_position` - Vector of positions to set.
    ///
    /// # Panics
    /// Panics if the particle or time_slice indices are out of bounds, or if the length of `bead_position`
    /// does not match the spatial dimension `d`.
    pub fn set_position(&mut self, particle: usize, time_slice: usize, bead_position: &[f64]) {
        // Ensure indices are within bounds
        assert!(
            particle < self.positions.shape()[0],
            "Particle index out of bounds: particle={}, max allowed={}",
            particle,
            self.positions.shape()[0] - 1
        );
        assert!(
            time_slice < self.positions.shape()[1],
            "Time slice index out of bounds: time_slice={}, max allowed={}",
            time_slice,
            self.positions.shape()[1] - 1
        );

        // Ensure the position vector matches the expected spatial dimension
        assert_eq!(
            bead_position.len(),
            self.positions.shape()[2],
            "Position length mismatch: expected={}, got={}",
            self.positions.shape()[2],
            bead_position.len()
        );

        // Efficiently set the position using the higher order `assign` function
        self.positions
            .slice_mut(s![particle, time_slice, ..])
            .assign(&arr1(bead_position));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_new_worldlines() {
        // Test valid initialization
        let world = WorldLines::new(2, 3, 3); // 2 particles, 3 time slices, 3D
        assert_eq!(world.positions.shape(), &[2, 3, 3]);

        // Test invalid dimensions (should panic)
        let result = std::panic::catch_unwind(|| WorldLines::new(0, 3, 3));
        assert!(result.is_err());
    }

    #[test]
    fn test_get_position() {
        let mut world = WorldLines::new(2, 3, 3);
        world.set_position(0, 0, &[1.0, 2.0, 3.0]);

        // Retrieve position
        let position = world.get_position(0, 0);
        assert_eq!(position, array![1.0, 2.0, 3.0]);

        // Test out-of-bounds access (should panic)
        let result = std::panic::catch_unwind(|| world.get_position(2, 0));
        assert!(result.is_err());
    }

    //#[ignore]
    #[test]
    fn test_get_position_mut() {
        let mut world = WorldLines::new(2, 3, 3);
        {
            // Modify position using mutable reference
            let mut position = world.get_position_mut(0, 0);
            position.assign(&array![4.0, 5.0, 6.0]);
        }

        // Verify the change
        let position = world.get_position(0, 0);
        assert_eq!(position, array![4.0, 5.0, 6.0]);
    }

    #[test]
    #[should_panic]
    fn test_invalid_arguments_get_position_mut() {
        let mut world = WorldLines::new(2, 3, 2);
        world.get_position_mut(1, 3);
    }

    #[test]
    fn test_combined_operations() {
        let mut world = WorldLines::new(1, 1, 2); // 1 particle, 1 time slice, 2D

        // Initial position should be zero
        let position = world.get_position(0, 0);
        assert_eq!(position, array![0.0, 0.0]);

        // Modify position
        world.set_position(0, 0, &[3.0, 4.0]);

        // Verify the modification
        let position = world.get_position(0, 0);
        assert_eq!(position, array![3.0, 4.0]);
    }
}
