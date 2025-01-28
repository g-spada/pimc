use super::traits::{BaseImage, Space};
//use crate::path_state::traits::{WorldLineDimensions, WorldLinePositionAccess};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

/// A struct to represent a box with periodic boundary conditions.
#[derive(Debug, PartialEq)]
pub struct PeriodicBox<const D: usize> {
    /// The lengths of the box in each spatial dimension.
    pub length: [f64; D],
}

impl<const D: usize> PeriodicBox<D> {
    /// Creates a new `PeriodicBox` instance.
    ///
    /// # Arguments
    /// * `length` - A vector specifying the length of the box in each spatial dimension.
    ///
    /// # Panics
    /// Panics if any value in `length` is less than or equal to zero.
    ///
    /// # Returns
    /// A new `PeriodicBox` instance with the specified length.
    pub fn new(length: [f64; D]) -> Self {
        assert!(
            length.iter().all(|&l| l > 0.0),
            "All box lengths must be positive."
        );
        Self { length }
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
            result[i] = diff - self.length[i] * (diff / self.length[i]).round();
        }
        result
    }

    /// Computes the Euclidean distance between two positions under periodic boundary conditions.
    ///
    /// This method calculates the periodic difference between two positions
    /// and then computes the Euclidean distance using the nearest-image convention.
    ///
    /// # Arguments
    /// * `r1` - The first position.
    /// * `r2` - The second position.
    ///
    /// # Returns
    /// The Euclidean distance between `r1` and `r2` under periodic boundary conditions.
    ///
    /// # Panics
    /// Panics if the dimensions of `r1`, `r2`, and the box lengths do not match.
    pub fn distance<'a, A, B>(&self, r1: A, r2: B) -> f64
    where
        A: Into<ArrayView1<'a, f64>>,
        B: Into<ArrayView1<'a, f64>>,
    {
        // Compute the periodic difference between the two positions.
        let diff = self.difference(r1, r2);

        // Compute the Euclidean distance as the square root of the sum of squared differences.
        diff.iter().map(|&d| d * d).sum::<f64>().sqrt()
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
            .zip(&self.length)
            .map(|(&x, &l)| x.rem_euclid(l))
            .collect()
    }
}

impl<const D: usize> Space for PeriodicBox<D> {
    const SPATIAL_DIMENSIONS: usize = D;

    /// D-dimensional volume of the space
    fn volume(&self) -> f64 {
        let mut volume = 1.0;
        for i in 0..D {
            volume *= self.length[i];
        }
        volume
    }

    /// Compute the vector difference between two points.
    fn difference<'a, A, B>(&self, r1: A, r2: B) -> Array1<f64>
    where
        A: Into<ArrayView1<'a, f64>>,
        B: Into<ArrayView1<'a, f64>>,
    {
        let r1_view = r1.into();
        let r2_view = r2.into();
        //(&r1_view - &r2_view).iter().zip(self.length.iter()).map(|(r,l)| r - l * (r/l).round()).collect()
        debug_assert_eq!(
            r1_view.len(),
            D,
            "Input arrays must have the correct dimensionality."
        );
        debug_assert_eq!(
            r2_view.len(),
            D,
            "Input arrays must have the correct dimensionality."
        );

        r1_view
            .iter()
            .zip(r2_view.iter())
            .zip(self.length.iter())
            .map(|((r1, r2), l)| {
                let diff = r1 - r2;
                diff - l * (diff / l).round()
            })
            .collect()
    }

    fn differences_from_reference<'a, A, B>(&self, r1: A, r2: B) -> Array2<f64>
    where
        A: Into<ArrayView2<'a, f64>>,
        B: Into<ArrayView1<'a, f64>>,
    {
        let r1_view = r1.into();
        let r2_view = r2.into();

        debug_assert_eq!(
            r1_view.shape()[1],
            D,
            "Each row of r1 must have the correct dimensionality."
        );
        debug_assert_eq!(
            r2_view.len(),
            D,
            "Reference vector must have the correct dimensionality."
        );

        let mut result = Array2::zeros(r1_view.dim());

        for (mut row, point) in result.outer_iter_mut().zip(r1_view.outer_iter()) {
            for i in 0..D {
                let diff = point[i] - r2_view[i];
                row[i] = diff - self.length[i] * (diff / self.length[i]).round();
            }
        }

        result
    }

    fn distance<'a, A, B>(&self, r1: A, r2: B) -> f64
    where
        A: Into<ArrayView1<'a, f64>>,
        B: Into<ArrayView1<'a, f64>>,
    {
        let r1_view = r1.into();
        let r2_view = r2.into();
        //(&r1_view - &r2_view).iter().zip(self.length.iter()).map(|(r,l)| r - l * (r/l).round()).collect()
        debug_assert_eq!(
            r1_view.len(),
            D,
            "Input arrays must have the correct dimensionality."
        );
        debug_assert_eq!(
            r2_view.len(),
            D,
            "Input arrays must have the correct dimensionality."
        );

        let diff: Array1<f64> = r1_view
            .iter()
            .zip(r2_view.iter())
            .zip(self.length.iter())
            .map(|((r1, r2), l)| {
                let diff = r1 - r2;
                diff - l * (diff / l).round()
            })
            .collect();

        diff.mapv(|x| x * x).sum().sqrt()
    }
}

impl<const D: usize> BaseImage for PeriodicBox<D> {
    fn base_image<'a, A>(&self, r: A) -> Array1<f64>
    where
        A: Into<ArrayView1<'a, f64>>,
    {
        let r_view = r.into();

        debug_assert_eq!(
            r_view.len(),
            D,
            "Input vector must have the correct dimensionality."
        );

        r_view
            .iter()
            .zip(self.length.iter())
            .map(|(coord, l)| coord.rem_euclid(*l))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use crate::space::periodic_box::PeriodicBox;
    use ndarray::array;

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
    fn test_periodic_box_distance() {
        // Define a periodic box with lengths for each dimension.
        let pbc = PeriodicBox::new([1.0, 2.0, 3.0]);

        // Test 1: Points without crossing boundaries.
        let r1 = array![0.4, 1.1, 1.4];
        let r2 = array![0.9, 0.1, 1.8];
        let dist = pbc.distance(&r1, &r2);

        // Expected distance calculated manually: sqrt(0.5^2 + 1.0^2 + 0.4^2) =~ 1.1874...
        let expected = (0.5f64.powi(2) + 1.0f64.powi(2) + 0.4f64.powi(2)).sqrt();
        assert!(
            (dist - expected).abs() < 1e-10,
            "Test 1 failed: distance = {:.10}",
            dist
        );

        // Test 2: Points crossing boundaries (wrap-around).
        let r3 = array![0.9, 1.9, 2.9];
        let r4 = array![0.1, 0.1, 0.1];
        let dist = pbc.distance(&r3, &r4);

        // Expected distance: Nearest-image convention
        // - For dimension 1: |0.9 - 0.1 - 1.0| = 0.2
        // - For dimension 2: Nearest image is |1.9 - 0.1 - 2.0| = 0.2
        // - For dimension 3: Nearest image is |2.9 - 0.1 - 3.0| = 0.2
        let expected = (0.2f64.powi(2) + 0.2f64.powi(2) + 0.2f64.powi(2)).sqrt();
        assert!(
            (dist - expected).abs() < 1e-10,
            "Test 2 failed: distance = {:.10}",
            dist
        );

        // Test 3: Identical points (distance should be zero).
        let r5 = array![0.5, 1.0, 1.5];
        let r6 = array![0.5, 1.0, 1.5];
        let dist = pbc.distance(&r5, &r6);

        assert!(
            (dist).abs() < 1e-10,
            "Test 3 failed: distance = {:.10}",
            dist
        );

        // Test 4: Periodic images of the same point (distance should be zero).
        let r7 = array![0.5, 1.0, 1.0];
        let r8 = array![-0.5, 5.0, -5.0];
        let dist = pbc.distance(&r7, &r8);

        assert!(
            (dist).abs() < 1e-10,
            "Test 4 failed: distance = {:.10}",
            dist
        );
    }
}
