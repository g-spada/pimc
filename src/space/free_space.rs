use ndarray::ArrayView1;

/// A struct to represent unrestricted Euclidean space in D dimensions.
#[derive(Debug)]
pub struct FreeSpace<const D: usize>;

impl<const D: usize> FreeSpace<D> {
    /// Computes the difference between two positions in free space.
    ///
    /// # Arguments
    /// * `r1` - The first position.
    /// * `r2` - The second position.
    ///
    /// # Returns
    /// A fixed-size array containing the difference between `r1` and `r2`
    /// for each dimension.
    ///
    /// # Panics
    /// Panics if the dimensions of `r1` and `r2` do not match `D`.
    pub fn difference<'a, A, B>(&self, r1: A, r2: B) -> [f64; D]
    where
        A: Into<ArrayView1<'a, f64>>,
        B: Into<ArrayView1<'a, f64>>,
    {
        let r1_view = r1.into();
        let r2_view = r2.into();

        // Ensure the input dimensions match
        debug_assert_eq!(
            r1_view.len(),
            r2_view.len(),
            "Arrays must have the same shape"
        );
        debug_assert_eq!(
            r1_view.len(),
            D,
            "Input array lengths must match the dimensionality of FreeSpace"
        );

        // Compute the element-wise difference
        let mut result = [0.0; D];
        for i in 0..r1_view.len() {
            result[i] = r1_view[i] - r2_view[i];
        }
        result
    }

    /// Computes the Euclidean distance between two positions in free space.
    ///
    /// # Arguments
    /// * `r1` - The first position.
    /// * `r2` - The second position.
    ///
    /// # Returns
    /// The Euclidean distance between `r1` and `r2`.
    ///
    /// # Panics
    /// Panics if the dimensions of `r1` and `r2` do not match `D`.
    pub fn distance<'a, A, B>(&self, r1: A, r2: B) -> f64
    where
        A: Into<ArrayView1<'a, f64>>,
        B: Into<ArrayView1<'a, f64>>,
    {
        let diff = self.difference(r1, r2);

        // Compute the Euclidean distance
        diff.iter().map(|&d| d * d).sum::<f64>().sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_free_space_difference() {
        let space = FreeSpace::<3>;

        let r1 = array![1.0, 2.0, 3.0];
        let r2 = array![4.0, 6.0, 8.0];

        let diff = space.difference(&r1, &r2);
        assert_eq!(diff, [-3.0, -4.0, -5.0]);
    }

    #[test]
    fn test_free_space_distance() {
        let space = FreeSpace::<3>;

        let r1 = array![1.0, 2.0, 3.0];
        let r2 = array![4.0, 6.0, 8.0];

        let dist = space.distance(&r1, &r2);

        // Expected distance: sqrt((-3)^2 + (-4)^2 + (-5)^2) = sqrt(50) = 7.071068
        assert!((dist - 7.071068).abs() < 1e-6);
    }
}
