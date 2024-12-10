use std::collections::{BTreeSet, HashMap};
use std::ops::Range;

/// A data structure that keeps track of the modified beads. It stores particle indices and ranges
/// of time-slices. This is a stripped down version of ProposedUpdate used as return type for Monte
/// Carlo updates.
///
/// The structure tracks:
/// - Modified bead ranges for each particle index as a collection of `Range<usize>`.
/// - The list of all modified particles.
/// - The union of all modified timeslices.
pub struct AcceptedUpdate {
    /// Maps particle indices to their respective ranges of modifications.
    pub modifications: HashMap<usize, Vec<Range<usize>>>,

    /// Cached set of all modified timeslices, stored in sorted order.
    pub modified_timeslices: BTreeSet<usize>,

    /// Cached set of all modified particle indices, stored in sorted order.
    pub modified_particles: BTreeSet<usize>,
}
