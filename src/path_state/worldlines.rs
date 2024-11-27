use super::particlestate::ParticleState;
use ndarray::{arr1, s, Array, Array1, Array2, Array3, ArrayView1, ArrayView2, ArrayViewMut1};
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
#[derive(Clone, Serialize, Deserialize)]
pub struct WorldLines<const N: usize, const M: usize, const D: usize> {
    /// Multidimensional array with const generics dimensions
    /// (N particles, M time slices, D spatial dimensions).
    positions: Array3<f64>,
    states: Array2<ParticleState>,
    prev_permutation: Array1<i32>, // -1 for open worldlines
    next_permutation: Array1<i32>, // -1 for open worldlines
    worm_head: Option<usize>,
    worm_tail: Option<usize>,
}

#[allow(clippy::new_without_default)]
impl<const N: usize, const M: usize, const D: usize> WorldLines<N, M, D> {
    /// Creates a new `WorldLines` instance with all positions initialized to zero.
    pub fn new() -> Self {
        Self {
            positions: Array::zeros((N, M, D)),
            states: Array::from_elem((N, M), ParticleState::Up),
            prev_permutation: Array1::from_iter((0..N).map(|i| i as i32)),
            next_permutation: Array1::from_iter((0..N).map(|i| i as i32)),
            worm_head: None,
            worm_tail: None,
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
        debug_assert!(
            particle < N,
            "Particle index out of bounds: particle={}, max allowed={}",
            particle,
            N - 1
        );
        debug_assert!(
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
        debug_assert!(
            particle < N,
            "Particle index out of bounds: particle={}, max allowed={}",
            particle,
            N - 1
        );
        debug_assert!(
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
        debug_assert!(
            particle < N,
            "Particle index out of bounds: particle={}, max allowed={}",
            particle,
            N - 1
        );
        debug_assert!(
            time_slice < M,
            "Time slice index out of bounds: time_slice={}, max allowed={}",
            time_slice,
            M - 1
        );

        // Ensure the position vector matches the expected spatial dimension
        debug_assert_eq!(
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
        debug_assert!(
            particle < N,
            "Particle index out of bounds: particle={}, max allowed={}",
            particle,
            N - 1
        );
        debug_assert!(
            start_slice < end_slice,
            "Invalid slice range: start_slice ({}) must be less than end_slice ({})",
            start_slice,
            end_slice
        );
        debug_assert!(
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
        debug_assert!(
            particle < N,
            "Particle index out of bounds: particle={}, max allowed={}",
            particle,
            N - 1
        );
        debug_assert!(
            start_slice < end_slice,
            "Invalid slice range: start_slice ({}) must be less than end_slice ({})",
            start_slice,
            end_slice
        );
        debug_assert!(
            end_slice <= M,
            "Time slice range out of bounds: end_slice={}, max allowed={}",
            end_slice,
            M
        );
        debug_assert_eq!(
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

    /// Gets the state of a specific particle at a specific time slice.
    ///
    /// # Arguments
    /// * `particle` - The index of the particle.
    /// * `time_slice` - The index of the time slice.
    ///
    /// # Returns
    /// The `ParticleState` of the specified particle at the given time slice.
    ///
    /// # Panics
    /// Panics if the `particle` or `time_slice` indices are out of bounds.
    pub fn get_particle_state(&self, particle: usize, time_slice: usize) -> ParticleState {
        debug_assert!(
            particle < N,
            "Particle index out of bounds: particle={}, max allowed={}",
            particle,
            N - 1
        );
        debug_assert!(
            time_slice < M,
            "Time slice index out of bounds: time_slice={}, max allowed={}",
            time_slice,
            M - 1
        );

        self.states[[particle, time_slice]]
    }

    /// Sets the state of a specific particle at a specific time slice.
    ///
    /// # Arguments
    /// * `particle` - The index of the particle.
    /// * `time_slice` - The index of the time slice.
    /// * `state` - The `ParticleState` to assign to the particle.
    ///
    /// # Panics
    /// Panics if the `particle` or `time_slice` indices are out of bounds.
    pub fn set_particle_state(&mut self, particle: usize, time_slice: usize, state: ParticleState) {
        debug_assert!(
            particle < N,
            "Particle index out of bounds: particle={}, max allowed={}",
            particle,
            N - 1
        );
        debug_assert!(
            time_slice < M,
            "Time slice index out of bounds: time_slice={}, max allowed={}",
            time_slice,
            M - 1
        );

        self.states[[particle, time_slice]] = state;
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

    /// Sets the previous particle permutation for a given particle.
    ///
    /// # Arguments
    /// * `particle` - The particle index to set.
    /// * `prev_particle` - `Some(index)` for a valid preceding particle or `None` for an open worldline.
    ///
    /// # Panics
    /// Panics if `particle` is out of bounds or if `prev_particle` is `Some(value)` and `value` is not in the valid range `0..N`.
    pub fn set_prev_permutation(&mut self, particle: usize, prev_particle: Option<usize>) {
        debug_assert!(
            particle < N,
            "Particle index out of bounds: particle={}, max allowed={}",
            particle,
            N - 1
        );

        match prev_particle {
            Some(index) => {
                debug_assert!(
                    index < N,
                    "Invalid prev_particle value: must be between 0 and N-1, got {}",
                    index
                );
                // Update the permutation
                self.prev_permutation[particle] = index as i32;
                // Update worm_tail as neccessary
                match self.worm_tail {
                    Some(tail) if tail == particle => self.worm_tail = None,
                    _ => (),
                }
            }
            None => {
                // Set tail
                self.prev_permutation[particle] = -1; // Open worldline
                self.worm_tail = Some(particle);
            }
        }
    }

    /// Sets the next particle permutation for a given particle.
    ///
    /// # Arguments
    /// * `particle` - The particle index to set.
    /// * `next_particle` - `Some(index)` for a valid preceding particle or `None` for an open worldline.
    ///
    /// # Panics
    /// Panics if `particle` is out of bounds or if `next_particle` is `Some(value)` and `value` is not in the valid range `0..N`.
    pub fn set_next_permutation(&mut self, particle: usize, next_particle: Option<usize>) {
        debug_assert!(
            particle < N,
            "Particle index out of bounds: particle={}, max allowed={}",
            particle,
            N - 1
        );

        match next_particle {
            Some(index) => {
                debug_assert!(
                    index < N,
                    "Invalid prev_particle value: must be between 0 and N-1, got {}",
                    index
                );
                // Update the permutation
                self.next_permutation[particle] = index as i32;
                // Update worm_head as neccessary
                match self.worm_head {
                    Some(head) if head == particle => self.worm_head = None,
                    _ => (),
                }
            }
            None => {
                // Set head
                self.next_permutation[particle] = -1; // Open worldline
                self.worm_head = Some(particle);
            }
        }
    }

    /// Gets the previous particle for a given particle.
    ///
    /// # Returns
    /// An `Option<usize>` where:
    /// * `Some(value)` indicates the preceding particle index.
    /// * `None` indicates an open worldline (-1).
    pub fn get_prev_permutation(&self, particle: usize) -> Option<usize> {
        debug_assert!(
            particle < N,
            "Particle index out of bounds: particle={}, max allowed={}",
            particle,
            N - 1
        );
        match self.prev_permutation[particle] {
            -1 => None,
            index => Some(index as usize),
        }
    }

    /// Gets the next particle for a given particle.
    ///
    /// # Returns
    /// An `Option<usize>` where:
    /// * `Some(value)` indicates the following particle index.
    /// * `None` indicates an open worldline (-1).
    pub fn get_next_permutation(&self, particle: usize) -> Option<usize> {
        debug_assert!(
            particle < N,
            "Particle index out of bounds: particle={}, max allowed={}",
            particle,
            N - 1
        );
        match self.next_permutation[particle] {
            -1 => None,
            index => Some(index as usize),
        }
    }

    /// Gets the worm head particle index, if it exists.
    ///
    /// # Returns
    /// An `Option<usize>`:
    /// * `Some(index)` - The index of the particle acting as the worm head.
    /// * `None` - If there is no worm head in the current configuration.
    pub fn get_worm_head(&self) -> Option<usize> {
        self.worm_head
    }

    /// Gets the worm tail particle index, if it exists.
    ///
    /// # Returns
    /// An `Option<usize>`:
    /// * `Some(index)` - The index of the particle acting as the worm head.
    /// * `None` - If there is no worm head in the current configuration.
    pub fn get_worm_tail(&self) -> Option<usize> {
        self.worm_tail
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
    fn test_particle_state() {
        const N: usize = 4;
        const M: usize = 8;
        const D: usize = 3;

        let mut world = WorldLines::<N, M, D>::new();

        // Default state should be Up
        for particle in 0..N {
            for time_slice in 0..M {
                assert_eq!(
                    world.get_particle_state(particle, time_slice),
                    ParticleState::Up
                );
            }
        }

        for time_slice in (0..M).step_by(2) {
            // Cange the particle state to Dn for particle 1 at even slices
            world.set_particle_state(1, time_slice, ParticleState::Dn)
        }

        for time_slice in 0..M {
            if time_slice % 2 == 0 {
                assert_eq!(world.get_particle_state(1, time_slice), ParticleState::Dn);
            } else {
                assert_eq!(world.get_particle_state(1, time_slice), ParticleState::Up);
            }
        }
    }

    #[test]
    fn test_set_permutations() {
        const NP: usize = 4;
        const NS: usize = 16;
        const ND: usize = 3;
        let mut world = WorldLines::<NP, NS, ND>::new();

        // Worldlines are initialized in the Z sector with all 1-cycle permutations
        for i in 0..NP {
            assert_eq!(world.get_prev_permutation(i), Some(i));
            assert_eq!(world.get_next_permutation(i), Some(i));
        }

        // Create a 2-cycle
        world.set_prev_permutation(0, Some(NP - 1));
        world.set_next_permutation(0, Some(NP - 1));
        world.set_prev_permutation(NP - 1, Some(0));
        world.set_next_permutation(NP - 1, Some(0));
        assert_eq!(world.get_prev_permutation(0), Some(NP - 1));
        assert_eq!(world.get_worm_tail(), None);
        assert_eq!(world.get_worm_head(), None);

        // Break the cycle
        world.set_prev_permutation(0, None);
        world.set_next_permutation(NP - 1, None);
        assert_eq!(world.get_worm_tail(), Some(0));
        assert_eq!(world.get_worm_head(), Some(NP - 1));
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
