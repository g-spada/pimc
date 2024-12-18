use ndarray::ArrayView1;

/// Trait for working with spatial boundaries and distances.
///
/// The `Space` trait provides methods to compute periodic differences,
/// Euclidean distances, and map positions to their fundamental image within a spatial domain.
pub trait Space<const D: usize> {
    ///// Returns the lengths of the space in each spatial dimension.
    //fn lengths(&self) -> &[f64; D];

    /// D-dimensional volume of the space
    fn volume(&self) -> f64;

    /// Computes the periodic difference between two positions.
    ///
    /// # Arguments
    /// * `r1` - The first position.
    /// * `r2` - The second position.
    ///
    /// # Returns
    /// A fixed-size array representing the periodic difference.
    fn difference<'a, A, B>(&self, r1: A, r2: B) -> [f64; D]
    where
        A: Into<ArrayView1<'a, f64>>,
        B: Into<ArrayView1<'a, f64>>;

    /// Computes the Euclidean distance between two positions under periodic boundary conditions.
    ///
    /// # Arguments
    /// * `r1` - The first position.
    /// * `r2` - The second position.
    ///
    /// # Returns
    /// The Euclidean distance.
    fn distance<'a, A, B>(&self, r1: A, r2: B) -> f64
    where
        A: Into<ArrayView1<'a, f64>>,
        B: Into<ArrayView1<'a, f64>>;
}
