use ndarray::{Array1,ArrayView1,Array2};

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

pub trait Space2 {
    const SPATIAL_DIMENSIONS: usize;

    /// Compute the vector difference between two points.
    fn difference<'a, A, B>(&self, r1: A, r2: B) -> Array1<f64>;

    /// Compute the vector differences to a reference point.
    fn differences_from_reference<'a, A, B>(&self, r1: A, r2: B) -> Array2<f64>;

    /// Compute the distance between two points.
    //fn distance(&self, p1: &[f64], p2: &[f64]) -> f64;
    fn distance<'a, A, B>(&self, r1: A, r2: B) -> f64;

    /// Get the point's base image within the fundamental simulation cell.
    //fn base_image(&self, point: &[f64]) -> &[f64];
    fn base_image<'a, A>(&self, r: A) -> Array1<f64>
    where
        A: Into<ArrayView1<'a, f64>>;
}
