use crate::path_state::traits::{WorldLineDimensions, WorldLinePositionAccess};
use ndarray::{Array1, ArrayView1};

/// A struct to represent a box with periodic boundary conditions.
///
/// The `PeriodicBox` struct provides utilities for working with periodic systems, such as
/// calculating the difference between positions under periodic boundary conditions, wrapping
/// positions to their fundamental image, and reseating polymers within the box.
#[derive(Debug, PartialEq)]
pub struct PeriodicBox<const D: usize> {
    /// The lengths of the box in each spatial dimension.
    lengths: [f64; D],
}

impl<const D: usize> PeriodicBox<D> {
    /// Creates a new `PeriodicBox` instance.
    ///
    /// # Arguments
    /// * `lengths` - A vector specifying the lengths of the box in each spatial dimension.
    ///
    /// # Returns
    /// A new `PeriodicBox` instance with the specified lengths.
    pub fn new(lengths: [f64; D]) -> Self {
        Self { lengths }
    }

    /// Computes the periodic difference between two positions.
    ///
    /// This function calculates the difference between two positions `r1` and `r2` and applies
    /// the nearest-image convention to ensure the result is in the range `[-length/2, length/2)`
    /// for each dimension of the box.
    ///
    /// # Arguments
    /// * `r1` - The first position.
    /// * `r2` - The second position.
    ///
    /// # Returns
    /// A fixed-size array containing the periodic difference between `r1` and `r2`,
    /// ensuring that the result is in the range `[-length/2, length/2]` for each dimension.
    ///
    /// # Panics
    /// Panics if the dimensions of `r1`, `r2`, and the box lengths do not match.
    pub fn difference<'a, A, B>(&self, r1: A, r2: B) -> [f64; D]
    where
        A: Into<ArrayView1<'a, f64>>,
        B: Into<ArrayView1<'a, f64>>,
    {
        let r1_view = r1.into();
        let r2_view = r2.into();

        // Ensure the shapes are the same
        debug_assert_eq!(
            r1_view.len(),
            r2_view.len(),
            "Arrays must have the same shape"
        );
        debug_assert_eq!(
            r1_view.len(),
            D,
            "Lengths vector must have the same length as input arrays"
        );

        // Compute the element-wise difference and apply the nearest-image transformation
        let mut result = [0.0; D];
        for i in 0..r1_view.len() {
            let diff = r1_view[i] - r2_view[i];
            result[i] = diff - self.lengths[i] * (diff / self.lengths[i]).round();
        }
        result
    }

    /// Maps a position to its fundamental image within the periodic box.
    ///
    /// This function ensures that a position `r` is wrapped into the range `[0, length)` for
    /// each dimension of the box, effectively bringing it to the fundamental image.
    ///
    /// # Arguments
    /// * `r` - The position as an array view.
    ///
    /// # Returns
    /// A 1D array containing the position wrapped into the fundamental image.
    ///
    /// # Panics
    /// This function will panic if:
    /// - The length of `r` does not match the number of box dimensions.
    pub fn fundamental_image<'a, A>(&self, r: A) -> Array1<f64>
    where
        A: Into<ArrayView1<'a, f64>>,
    {
        r.into()
            .iter()
            .zip(&self.lengths)
            .map(|(&x, &l)| x.rem_euclid(l))
            .collect()
    }

    /// Reseats a polymer within the periodic box.
    ///
    /// This method adjusts all beads of a polymer such that the first bead is mapped to its
    /// fundamental image, while maintaining the relative configuration of the polymer.
    ///
    /// # Arguments
    /// * `worldlines` - A mutable reference to the worldlines object containing particle positions.
    /// * `particle` - The index of the particle (polymer) to reseat.
    ///
    /// # Behavior
    /// Adjusts all positions of the specified particle within the periodic box, ensuring
    /// the polymer is aligned with the fundamental image of the box.
    ///
    /// # Panics
    /// This function will panic if:
    /// - The particle index is out of bounds.
    pub fn reseat_polymer<W>(&self, worldlines: &mut W, particle: usize)
    where
        W: WorldLineDimensions + WorldLinePositionAccess,
    {
        // Validate input
        debug_assert!(particle < worldlines.particles(), "Invalid particle index");
        debug_assert_eq!(
            D,
            worldlines.spatial_dimensions(),
            "Spatial dimensions don't match"
        );

        // Compute the fundamental image of the first bead
        let first_image = self.fundamental_image(worldlines.position(particle, 0));

        // Compute the shift
        let shift = &first_image - &worldlines.position(particle, 0);

        // Extract the total time slices and the current positions
        let total_time_slices = worldlines.time_slices();

        let mut whole_polymer = worldlines.positions_mut(particle, 0, total_time_slices);

        // Add the shift to each bead of the polymer
        whole_polymer += &shift.view().broadcast([total_time_slices, D]).unwrap();
    }
}

#[test]
fn test_periodic_box_difference() {
    use ndarray::array;
    let pbc = PeriodicBox::new([1.0, 1.0, 2.0]);
    let diff = pbc.difference(&array![0.4, 1.1, 1.8], &array![0.0, 0.0, 0.0]);
    let expected = array![0.4, 0.1, -0.2];
    for i in 0..3 {
        assert!((diff[i] - expected[i]).abs() < 1e-15);
    }
}

#[test]
fn test_periodic_box_fundamental_image() {
    use ndarray::array;
    let pbc = PeriodicBox::new([1.0, 2.0, 4.0]);
    let image = pbc.fundamental_image(&array![0.6, -3.1, 10.8]);
    let expected = array![0.6, 0.9, 2.8];
    for i in 0..3 {
        assert!((image[i] - expected[i]).abs() < 1e-15);
    }
    let image = pbc.fundamental_image(&array![-3.0, 0.0, 8.0]);
    for i in 0..3 {
        assert!((image[i] - 0.0).abs() < 1e-15);
    }
}

#[test]
fn test_reseat_polymer() {
    use crate::path_state::worm::Worm;
    use ndarray::{array, Zip};

    // Define a periodic box
    let pbc = PeriodicBox::new([1.0, 1.0, 1.0]); // Box of length 1.0 in all dimensions

    // Polymer object
    let mut worldlines = Worm::<1, 4, 3>::new();

    // Initial polymer positions: Particle 0, slices 0-2
    let initial_positions = array![[1.2, -0.8, 0.5], [1.3, -0.9, 0.4], [1.1, -1.1, 0.6], [0.0, 0.0, 0.0]];

    worldlines.set_positions(0, 0, 4, &initial_positions);

    // Reseat the polymer for particle 0
    pbc.reseat_polymer(&mut worldlines, 0);

    // Expected positions after reseating
    let expected_positions = array![
        [0.2, 0.2, 0.5],
        [0.3, 0.1, 0.4],
        [0.1, -0.1, 0.6],
        [-1.0, 1.0, 0.0]
    ];

    // Assert that the positions match the expected values
    let result = worldlines.positions(0, 0, 4);
    Zip::from(&expected_positions)
        .and(result)
        .for_each(|&val1, &val2| assert!((val1 - val2).abs() < 1e-15));
}
