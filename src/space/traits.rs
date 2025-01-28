use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

/// Trait for working with spatial boundaries and distances.
pub trait Space {
    const SPATIAL_DIMENSIONS: usize;

    /// D-dimensional volume of the space
    fn volume(&self) -> f64;

    /// Compute the vector difference between two points.
    fn difference<'a, A, B>(&self, r1: A, r2: B) -> Array1<f64>
    where
        A: Into<ArrayView1<'a, f64>>,
        B: Into<ArrayView1<'a, f64>>;

    /// Compute the vector differences to a reference point.
    fn differences_from_reference<'a, A, B>(&self, r1: A, r2: B) -> Array2<f64>
    where
        A: Into<ArrayView2<'a, f64>>,
        B: Into<ArrayView1<'a, f64>>;

    /// Compute the distance between two points.
    fn distance<'a, A, B>(&self, r1: A, r2: B) -> f64
    where
        A: Into<ArrayView1<'a, f64>>,
        B: Into<ArrayView1<'a, f64>>;
}

pub trait BaseImage {
    /// Get the point's base image within the fundamental simulation cell.
    fn base_image<'a, A>(&self, r: A) -> Array1<f64>
    where
        A: Into<ArrayView1<'a, f64>>;
}
