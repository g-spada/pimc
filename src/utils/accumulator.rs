/// A structure for computing statistics using the blocking method.
///
/// The blocking method reduces the effects of autocorrelation in Monte Carlo
/// simulations by grouping consecutive measurements into blocks. Each block
/// stores the sum of its values and their squares, enabling the computation
/// of mean, variance, error of the mean, and autocorrelation time.
///
/// # Generics
/// - `N_BLOCKS`: Number of blocks to use (must be a power of 2).
/// - `INIT_BLOCK_DEPTH`: Initial depth of each block (number of measurements per block).
///
/// # Methods
/// - `add`: Adds a measurement to the accumulator.
/// - `statistics`: Computes mean, error, autocorrelation time, and variance.
/// - `clear`: Resets the accumulator to its initial state.
pub struct Accumulator<const N_BLOCKS: usize = 1024, const INIT_BLOCK_DEPTH: u64 = 16> {
    /// Total number of input values added.
    total_inputs: u64,

    /// Current number of measurements per block.
    block_depth: u64,

    /// Index of the currently active block.
    active_block: usize,

    /// Number of measurements added to the current block.
    block_filling: u64,

    /// Array storing the sum of values in each block.
    blocks: [f64; N_BLOCKS],

    /// Array storing the sum of squared values in each block.
    blocks_sq: [f64; N_BLOCKS],
}

impl<const N_BLOCKS: usize, const INIT_BLOCK_DEPTH: u64> Default
    for Accumulator<N_BLOCKS, INIT_BLOCK_DEPTH>
{
    /// `Default` implementation for `Accumulator<N_BLOCKS, INIT_BLOCK_DEPTH>`
    fn default() -> Self {
        Self::new()
    }
}

impl<const N_BLOCKS: usize, const INIT_BLOCK_DEPTH: u64> Accumulator<N_BLOCKS, INIT_BLOCK_DEPTH> {
    /// Creates a new `Accumulator` instance.
    ///
    /// # Panics
    /// Panics if `N_BLOCKS` is not a power of 2.
    pub fn new() -> Self {
        assert!(N_BLOCKS.is_power_of_two(), "N_BLOCKS must be a power of 2");
        Self {
            total_inputs: 0,
            block_depth: INIT_BLOCK_DEPTH,
            active_block: 0,
            block_filling: 0,
            blocks: [0.0; N_BLOCKS],
            blocks_sq: [0.0; N_BLOCKS],
        }
    }

    /// Returns the total number of input values added to the accumulator.
    pub fn size(&self) -> u64 {
        self.total_inputs
    }

    /// Adds a new measurement to the accumulator.
    ///
    /// If the current block is full, it moves to the next block. When all blocks
    /// are filled, the block depth is doubled, and the data is consolidated into
    /// fewer, larger blocks.
    pub fn add(&mut self, val: f64) {
        let n_blocks_half: usize = N_BLOCKS / 2;

        // Add the value to the current block
        self.blocks[self.active_block] += val;
        self.blocks_sq[self.active_block] += val * val;
        self.total_inputs += 1;
        self.block_filling += 1;

        // Check if the current block is full
        if self.block_filling == self.block_depth {
            self.block_filling = 0;
            self.active_block += 1;

            // If all blocks are full, consolidate and double the block depth
            if self.active_block == N_BLOCKS {
                self.active_block = n_blocks_half;
                self.block_depth *= 2;

                // Merge blocks in pairs
                for i in 0..n_blocks_half {
                    self.blocks[i] = self.blocks[2 * i] + self.blocks[2 * i + 1];
                    self.blocks_sq[i] = self.blocks_sq[2 * i] + self.blocks_sq[2 * i + 1];
                }

                // Reset the remaining blocks
                for i in n_blocks_half..N_BLOCKS {
                    self.blocks[i] = 0.0;
                    self.blocks_sq[i] = 0.0;
                }
            }
        }
    }

    /// Computes and returns the mean, error of the mean, autocorrelation time, and standard deviation.
    ///
    /// Returns `None` if no measurements have been added.
    ///
    /// # Returns
    /// A tuple containing:
    /// - `mean`: The mean of the measurements.
    /// - `mean_error`: The error of the mean, accounting for autocorrelation.
    /// - `autocorr_time`: The estimated autocorrelation time.
    /// - `std_dev`: The square root of the variance (standard deviation).
    pub fn statistics(&self) -> Option<(f64, f64, f64, f64)> {
        if self.total_inputs == 0 {
            return None;
        }

        // Compute the mean of the sample data
        let mean =
            self.blocks.iter().take(self.active_block + 1).sum::<f64>() / self.total_inputs as f64;
        let variance = -mean * mean
            + self
                .blocks_sq
                .iter()
                .take(self.active_block + 1)
                .sum::<f64>()
                / self.total_inputs as f64;

        // Compute the error of the mean
        let mut error_square = 0.0;
        for block_sum in self.blocks.iter().take(self.active_block) {
            let block_mean = block_sum / self.block_depth as f64;
            let diff_j = block_mean - mean;
            error_square += diff_j * diff_j;
        }
        let filled_blocks_weight = self.block_depth as f64 / self.total_inputs as f64;
        error_square *= filled_blocks_weight * filled_blocks_weight;
        // Include the error contribution from the partially filled block
        if self.block_filling > 0 {
            let active_block_mean = self.blocks[self.active_block] / self.block_filling as f64;
            let weight_last = self.block_filling as f64 / self.total_inputs as f64;
            let diff_last = weight_last * (active_block_mean - mean);
            error_square += diff_last * diff_last;
        }

        // Compute autocorrelation time
        let autocorr_time = self.total_inputs as f64 * error_square / variance;

        Some((mean, error_square.sqrt(), autocorr_time, variance.sqrt()))
    }

    /// Resets the accumulator, clearing all measurements and restoring initial settings.
    pub fn clear(&mut self) {
        self.total_inputs = 0;
        self.block_depth = INIT_BLOCK_DEPTH;
        self.active_block = 0;
        self.block_filling = 0;
        self.blocks.fill(0.0);
        self.blocks_sq.fill(0.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_accumulator() {
        let mut acc = Accumulator::<1024, 16>::new();

        // Add 1000 measurements
        for i in 0..1000 {
            acc.add(i as f64);
        }

        let stats = acc.statistics().unwrap();
        println!("Mean: {}", stats.0);
        println!("Mean Error: {}", stats.1);
        println!("Autocorrelation Time: {}", stats.2);
        println!("Standard Deviation: {}", stats.3);

        // Validate the mean is within a reasonable range
        assert!(stats.0 > 490.0 && stats.0 < 510.0);
    }

    #[test]
    fn test_accumulator_small_input() {
        // Input measurements
        let measurements = [1.0, 2.0, 3.0, 4.0, 5.0];
        let n_measurements = measurements.len() as f64;

        // Expected mean
        let expected_mean = measurements.iter().sum::<f64>() / n_measurements;

        // Expected variance
        let mean_square = measurements.iter().map(|&x| x * x).sum::<f64>() / n_measurements;
        let expected_variance = mean_square - expected_mean * expected_mean;

        // Expected standard deviation
        let expected_std_dev = expected_variance.sqrt();

        // Initialize the accumulator
        let mut acc = Accumulator::<16, 4>::new();

        // Add measurements to the accumulator
        for &value in &measurements {
            acc.add(value);
        }

        // Get statistics from the accumulator
        let (mean, _mean_error, _autocorr_time, std_dev) = acc.statistics().unwrap();

        // Assert the mean is as expected
        assert!(
            (mean - expected_mean).abs() < 1e-10,
            "Mean mismatch: got {}, expected {}",
            mean,
            expected_mean
        );

        // Assert the standard deviation is as expected
        assert!(
            (std_dev - expected_std_dev).abs() < 1e-10,
            "Standard deviation mismatch: got {}, expected {}",
            std_dev,
            expected_std_dev
        );
    }

    #[test]
    fn test_accumulator_with_reblocking() {
        // Number of blocks and block depth
        const N_BLOCKS: usize = 16;
        const INIT_BLOCK_DEPTH: u64 = 4;

        // Generate a large dataset (larger than N_BLOCKS * INIT_BLOCK_DEPTH)
        let measurements: Vec<f64> = (1..=1000).map(|x| x as f64).collect();
        let n_measurements = measurements.len() as f64;

        // Expected mean
        let expected_mean = measurements.iter().sum::<f64>() / n_measurements;

        // Expected variance
        let mean_square = measurements.iter().map(|&x| x * x).sum::<f64>() / n_measurements;
        let expected_variance = mean_square - expected_mean * expected_mean;

        // Expected standard deviation
        let expected_std_dev = expected_variance.sqrt();

        // Initialize the accumulator
        let mut acc = Accumulator::<N_BLOCKS, INIT_BLOCK_DEPTH>::new();

        // Add measurements to the accumulator
        for &value in &measurements {
            acc.add(value);
        }

        // Get statistics from the accumulator
        let (mean, _mean_error, _autocorr_time, std_dev) = acc.statistics().unwrap();

        // Assert the mean is as expected
        assert!(
            (mean - expected_mean).abs() < 1e-10,
            "Mean mismatch: got {}, expected {}",
            mean,
            expected_mean
        );

        // Assert the standard deviation is as expected
        assert!(
            (std_dev - expected_std_dev).abs() < 1e-10,
            "Standard deviation mismatch: got {}, expected {}",
            std_dev,
            expected_std_dev
        );
    }
}
