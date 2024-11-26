use ndarray::{arr1, s, Array, Array2, ArrayView1, ArrayView2, ArrayViewMut1, Dim};
use serde::{Deserialize, Serialize};
use serde_json;
use std::fs::File;
use std::io::{self, BufReader, BufWriter};

/// WorldLines struct to store particle positions using ndarray.
///
/// # Example
/// ```
/// use pimc_rs::path_state::worldlines::WorldLines;
///
/// // Create a new WorldLines instance
/// let mut world = WorldLines::<2,3,3>::new(); // 2 particles, 3 time slices, 3D space
///
/// // Set the position of the first particle at the first time slice
/// world.set_position(0, 0, &[1.0, 2.0, 3.0]);
///
/// // Retrieve the position
/// let position = world.get_position(0, 0);
/// assert_eq!(position.to_vec(), vec![1.0, 2.0, 3.0]);
/// ```
///
#[derive(Serialize, Deserialize)]
pub struct WorldLines<const N: usize, const M: usize, const D: usize> {
    /// Multidimensional array with const generics dimensions
    /// (N particles, M time slices, D spatial dimensions).
    positions: Array<f64, Dim<[usize; 3]>>,
}

impl<const N: usize, const M: usize, const D: usize> WorldLines<N, M, D> {
    /// Creates a new `WorldLines` instance with all positions initialized to zero.
    pub fn new() -> Self {
        Self {
            positions: Array::zeros((N, M, D)),
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
            particle < N,
            "Particle index out of bounds: particle={}, max allowed={}",
            particle,
            N - 1
        );
        assert!(
            time_slice < M,
            "Time slice index out of bounds: time_slice={}, max allowed={}",
            time_slice,
            M - 1
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
            particle < N,
            "Particle index out of bounds: particle={}, max allowed={}",
            particle,
            N - 1
        );
        assert!(
            time_slice < M,
            "Time slice index out of bounds: time_slice={}, max allowed={}",
            time_slice,
            M - 1
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
            particle < N,
            "Particle index out of bounds: particle={}, max allowed={}",
            particle,
            N - 1
        );
        assert!(
            time_slice < M,
            "Time slice index out of bounds: time_slice={}, max allowed={}",
            time_slice,
            M - 1
        );

        // Ensure the position vector matches the expected spatial dimension
        assert_eq!(
            bead_position.len(),
            D,
            "Position length mismatch: expected={}, got={}",
            D,
            bead_position.len()
        );

        // Efficiently set the position using the higher order `assign` function
        self.positions
            .slice_mut(s![particle, time_slice, ..])
            .assign(&arr1(bead_position));
    }

    /// Gets a 2D view of the positions for a specific particle across a range of time slices.
    ///
    /// # Arguments
    /// * `particle` - Index of the particle.
    /// * `start_slice` - Starting index of the time slice (inclusive).
    /// * `end_slice` - Ending index of the time slice (exclusive).
    ///
    /// # Returns
    /// A 2D array view of the positions for the specified particle and time slice range.
    ///
    /// # Panics
    /// Panics if:
    /// - The `particle` index is out of bounds.
    /// - The `start_slice` or `end_slice` indices are out of bounds.
    /// - `start_slice >= end_slice`.
    ///
    /// # Example
    /// ```
    /// use pimc_rs::path_state::worldlines::WorldLines;
    /// use ndarray::array;
    ///
    /// let mut world = WorldLines::<2,5,3>::new(); // 2 particles, 5 time slices, 3D space
    /// world.set_position(0, 0, &[1.0, 2.0, 3.0]);
    /// world.set_position(0, 1, &[4.0, 5.0, 6.0]);
    ///
    /// // Retrieve positions from slice 0 to 2 (exclusive)
    /// let positions = world.get_positions(0, 0, 2);
    /// assert_eq!(positions, array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    /// ```
    pub fn get_positions(
        &self,
        particle: usize,
        start_slice: usize,
        end_slice: usize,
    ) -> ArrayView2<f64> {
        assert!(
            particle < N,
            "Particle index out of bounds: particle={}, max allowed={}",
            particle,
            N - 1
        );
        assert!(
            start_slice < end_slice,
            "Invalid slice range: start_slice ({}) must be less than end_slice ({})",
            start_slice,
            end_slice
        );
        assert!(
            end_slice <= M,
            "Time slice range out of bounds: end_slice={}, max allowed={}",
            end_slice,
            M
        );

        // Extract and return a 2D view using slicing
        self.positions
            .slice(s![particle, start_slice..end_slice, ..])
    }

    /// Sets the positions for a specific particle across a range of time slices.
    ///
    /// # Arguments
    /// * `particle` - Index of the particle.
    /// * `start_slice` - Starting index of the time slice (inclusive).
    /// * `end_slice` - Ending index of the time slice (exclusive).
    /// * `positions` - A 2D array with shape `(end_slice - start_slice, spatial_dim)` containing the new positions.
    ///
    /// # Panics
    /// Panics if:
    /// - The `particle` index is out of bounds.
    /// - The `start_slice` or `end_slice` indices are out of bounds.
    /// - The `positions` array shape does not match `(end_slice - start_slice, spatial_dim)`.
    ///
    /// # Example
    /// ```
    /// use pimc_rs::path_state::worldlines::WorldLines;
    /// use ndarray::array;
    ///
    /// let mut world = WorldLines::<2,5,3>::new(); // 2 particles, 5 time slices, 3D space
    ///
    /// // Set positions for slices 0 to 2 (exclusive) for particle 0
    /// let new_positions = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
    /// world.set_positions(0, 0, 2, &new_positions);
    ///
    /// // Verify the changes
    /// let positions = world.get_positions(0, 0, 2);
    /// assert_eq!(positions, new_positions);
    /// ```
    pub fn set_positions(
        &mut self,
        particle: usize,
        start_slice: usize,
        end_slice: usize,
        positions: &Array2<f64>,
    ) {
        assert!(
            particle < N,
            "Particle index out of bounds: particle={}, max allowed={}",
            particle,
            N - 1
        );
        assert!(
            start_slice < end_slice,
            "Invalid slice range: start_slice ({}) must be less than end_slice ({})",
            start_slice,
            end_slice
        );
        assert!(
            end_slice <= M,
            "Time slice range out of bounds: end_slice={}, max allowed={}",
            end_slice,
            M
        );
        assert_eq!(
            positions.shape(),
            &[end_slice - start_slice, D],
            "Input positions shape mismatch: expected ({}, {}), got {:?}",
            end_slice - start_slice,
            D,
            positions.shape()
        );

        // Perform bulk assignment using slice and `assign`
        self.positions
            .slice_mut(s![particle, start_slice..end_slice, ..])
            .assign(positions);
    }

    /// Initializes all slices of each particle to the same position vector,
    /// generated by the input function.
    ///
    /// # Arguments
    /// * `position_generator` - A function or closure that takes a particle index
    ///   and returns a position vector of length `D`.
    ///
    /// # Panics
    /// Panics if the generated position vector does not match the spatial dimension `D`.
    ///
    /// # Example
    /// ```
    /// use pimc_rs::path_state::worldlines::WorldLines;
    /// use ndarray::array;
    ///
    /// let mut world = WorldLines::<2, 3, 3>::new();
    ///
    /// // Initialize positions with a simple function
    /// world.initialize_positions(|particle| vec![particle as f64; 3]);
    ///
    /// // Verify initialization
    /// assert_eq!(world.get_position(0, 0), array![0.0, 0.0, 0.0]);
    /// assert_eq!(world.get_position(1, 1), array![1.0, 1.0, 1.0]);
    /// ```
    pub fn initialize_positions<F>(&mut self, position_generator: F)
    where
        F: Fn(usize) -> Vec<f64>,
    {
        for particle in 0..N {
            // Generate the position vector for the current particle
            let position = position_generator(particle);
            assert_eq!(
                position.len(),
                D,
                "Generated position length mismatch: expected={}, got={}",
                D,
                position.len()
            );

            // Set all slices for the current particle to the same position vector
            for time_slice in 0..M {
                self.set_position(particle, time_slice, &position);
            }
        }
    }

    /// Saves the WorldLines instance to a file in JSON format.
    ///
    /// This method serializes the `WorldLines` instance into a human-readable JSON
    /// format. It is useful for saving the state of the simulation for later analysis.
    ///
    /// # Arguments
    /// * `filename` - Path to the output JSON file.
    ///
    /// # Errors
    /// Returns an error if the file cannot be created or written to.
    ///
    pub fn save_to_file(&self, filename: &str) -> io::Result<()> {
        let file = File::create(filename)?;
        let writer = BufWriter::new(file);
        serde_json::to_writer(writer, &self)?;
        Ok(())
    }

    /// Loads a WorldLines instance from a JSON file.
    ///
    /// This method deserializes a JSON file back into a `WorldLines` instance.
    /// It is useful for restoring a saved simulation state.
    ///
    /// # Arguments
    /// * `filename` - Path to the input JSON file.
    ///
    /// # Errors
    /// Returns an error if the file cannot be read or is invalid.
    ///
    pub fn load_from_file(filename: &str) -> io::Result<Self> {
        let file = File::open(filename)?;
        let reader = BufReader::new(file);
        let world_lines = serde_json::from_reader(reader)?;
        Ok(world_lines)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_new_worldlines() {
        // Test valid initialization
        let world = WorldLines::<2, 3, 3>::new(); // 2 particles, 3 time slices, 3D
        assert_eq!(world.positions.shape(), &[2, 3, 3]);
    }

    #[test]
    fn test_get_position() {
        let mut world = WorldLines::<2, 3, 3>::new();
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
        let mut world = WorldLines::<2, 3, 3>::new();
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
        let mut world = WorldLines::<2, 3, 2>::new();
        world.get_position_mut(1, 3);
    }

    #[test]
    fn test_combined_operations() {
        let mut world = WorldLines::<1, 1, 2>::new(); // 1 particle, 1 time slice, 2D

        // Initial position should be zero
        let position = world.get_position(0, 0);
        assert_eq!(position, array![0.0, 0.0]);

        // Modify position
        world.set_position(0, 0, &[3.0, 4.0]);

        // Verify the modification
        let position = world.get_position(0, 0);
        assert_eq!(position, array![3.0, 4.0]);
    }
    #[test]
    fn test_get_positions() {
        let mut world = WorldLines::<2, 5, 3>::new(); // 2 particles, 5 time slices, 3D space

        // Set positions for particle 0, slices 0 to 2
        world.set_position(0, 0, &[1.0, 2.0, 3.0]);
        world.set_position(0, 1, &[4.0, 5.0, 6.0]);

        // Get positions for slices 0 to 2 (exclusive) for particle 0
        let positions = world.get_positions(0, 0, 2);
        assert_eq!(positions, array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    }

    #[test]
    fn test_set_positions() {
        let mut world = WorldLines::<2, 5, 3>::new(); // 2 particles, 5 time slices, 3D space

        // Create a 2D array with new positions for slices 0 to 2
        let new_positions = array![
            [1.0, 2.0, 3.0], // Slice 0
            [4.0, 5.0, 6.0], // Slice 1
        ];

        // Set positions for particle 0, slices 0 to 2
        world.set_positions(0, 0, 2, &new_positions);

        // Verify the changes using get_positions
        let positions = world.get_positions(0, 0, 2);
        assert_eq!(positions, new_positions);
    }

    #[test]
    fn test_save_and_load_json_temp() -> io::Result<()> {
        let mut world = WorldLines::<2, 3, 3>::new();
        world.set_position(0, 0, &[1.0, 2.0, 3.0]);

        // Use a temporary file for saving
        use tempfile::NamedTempFile;
        let temp_file = NamedTempFile::new()?;
        world.save_to_file(temp_file.path().to_str().unwrap())?;

        // Load from the temporary file
        let loaded_world =
            WorldLines::<2, 3, 3>::load_from_file(temp_file.path().to_str().unwrap())?;

        // Verify the data
        assert_eq!(loaded_world.get_position(0, 0), array![1.0, 2.0, 3.0]);

        Ok(())
    }
}
