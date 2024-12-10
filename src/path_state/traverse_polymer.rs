use super::traits::{WorldLinePermutationAccess, WorldLineWormAccess};

/// Traverses a polymer to determine its configuration and find the last particle in the chain.
///
/// This method starts at a given particle index (`start`) and traverses the polymer chain
/// using the `preceding` connections in the worldlines. The traversal continues until either:
/// - A closed cycle is detected (i.e., the traversal returns to the starting particle).
/// - An open end of the polymer (a worm tail) is reached.
///
/// # Arguments
/// * `worldlines` - A reference to the worldlines object implementing the required traits.
/// * `start` - The starting particle index for the traversal.
///
/// # Returns
/// * The particle index where the traversal ends:
///   - If a closed cycle is detected, it returns the starting particle index.
///   - If an open polymer is encountered, it returns the particle index of the worm tail.
///
/// # Panics
/// This method will panic if the traversal does not reach the worm tail in the case of an open
/// polymer, indicating a logical inconsistency in the polymer's structure.
///
/// # Debugging
/// - Debug assertions ensure that if the traversal ends at an open polymer, the final particle
///   matches the worm tail in the worldlines.
pub fn traverse_polymer<W>(worldlines: &W, start: usize) -> usize
where
    W: WorldLinePermutationAccess + WorldLineWormAccess,
{
    let mut current = start;
    while let Some(prev) = worldlines.preceding(current) {
        if prev == start {
            return start; // Closed cycle
        }
        current = prev;
    }
    debug_assert_eq!(
        Some(current),
        worldlines.worm_tail(),
        "Expected to reach the worm tail for an open polymer"
    );
    current
}
