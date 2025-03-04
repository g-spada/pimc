/// Tracks attempts and accepted steps for an update.
#[derive(Debug, Clone, Copy)]
pub struct UpdateStats {
    pub attempts: usize,
    pub accepted: usize,
}

impl UpdateStats {
    /// Create a new UpdateStats instance with zeroed counters.
    pub fn new() -> Self {
        Self {
            attempts: 0,
            accepted: 0,
        }
    }

    /// Reset both counters to zero.
    pub fn reset(&mut self) {
        self.attempts = 0;
        self.accepted = 0;
    }
}

impl Default for UpdateStats {
    fn default() -> Self {
        Self::new()
    }
}
