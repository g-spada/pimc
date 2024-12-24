use crate::space::traits::Space2;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

pub struct FreeSpace<const D: usize>;

impl<const D: usize> Space2 for FreeSpace<D> {
    const SPATIAL_DIMENSIONS: usize = D;

    fn volume(&self) -> f64 {
        // FOR TEST PURPOSES ONLY
        1.0
    }

    fn difference<'a, A, B>(&self, r1: A, r2: B) -> Array1<f64>
    where
        A: Into<ArrayView1<'a, f64>>,
        B: Into<ArrayView1<'a, f64>>,
    {
        let r1_view = r1.into();
        let r2_view = r2.into();

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

        &r1_view - &r2_view
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

        // Subtract the reference point from all rows in r1
        let reference_broadcasted = r2_view.insert_axis(ndarray::Axis(0));
        r1_view.to_owned() - reference_broadcasted
    }

    fn distance<'a, A, B>(&self, r1: A, r2: B) -> f64
    where
        A: Into<ArrayView1<'a, f64>>,
        B: Into<ArrayView1<'a, f64>>,
    {
        let diff = self.difference(r1, r2);
        diff.mapv(|x| x * x).sum().sqrt()
    }

    fn base_image<'a, A>(&self, r: A) -> Array1<f64>
    where
        A: Into<ArrayView1<'a, f64>>,
    {
        // For FreeSpace, no periodic wrapping is needed, so simply return the input as is
        r.into().to_owned()
    }
}
