use ndarray::{Array3, ArrayView1, ArrayViewMut1, s};

/// WorldLines struct to store particle positions using ndarray.
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
    pub fn get_position(&self, particle: usize, time_slice: usize) -> ArrayView1<f64> {
        self.positions.slice(s![particle, time_slice, ..])
    }

    /// Gets a mutable view of the position of a specific particle at a specific time slice.
    /// # Arguments
    /// * `particle` - Index of the particle.
    /// * `time_slice` - Index of the time slice.
    ///
    /// # Returns
    /// An mutable array view of the positions.
    pub fn get_position_mut(&mut self, particle: usize, time_slice: usize) -> ArrayViewMut1<f64> {
        self.positions.slice_mut(s![particle, time_slice, ..])
    }

    /// Sets the position of a specific particle at a specific time slice.
    ///
    /// # Arguments
    /// * `particle` - Index of the particle.
    /// * `time_slice` - Index of the time slice.
    /// * `bead_position` - Vector of positions to set.
    ///
    /// # Returns
    /// Sets the position of a specific time_slice in a particle
    pub fn set_position(&mut self, particle: usize, time_slice: usize, bead_position: &[f64]) {
        assert_eq!(bead_position.len(), self.positions.shape()[2], "Position must match dimension d");
        for (i, &val) in bead_position.iter().enumerate() {
            self.positions[[particle, time_slice, i]] = val;
        }
    }
}

