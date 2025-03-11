use super::accepted_update::AcceptedUpdate;
use super::monte_carlo_update::MonteCarloUpdate;
use crate::action::traits::PotentialDensityMatrix;
use crate::system::traits::SystemAccess;
use rand::distributions::WeightedIndex;
use rand::prelude::Distribution;

pub struct PathIntegralMarkovChain<S, A, R> {
    pub system: S,
    pub action: A,
    updates: Vec<Box<dyn MonteCarloUpdate<S, A, R>>>,
    weights: Vec<f64>,
    rng: R,
    //step_count: usize,
    weighted_index: WeightedIndex<f64>,
}

impl<S, A, R> PathIntegralMarkovChain<S, A, R>
where
    S: SystemAccess,
    A: PotentialDensityMatrix,
    R: rand::Rng,
{
    pub fn new(
        system: S,
        action: A,
        updates: Vec<Box<dyn MonteCarloUpdate<S, A, R>>>,
        weights: Vec<f64>,
        rng: R,
    ) -> Self {
        assert_eq!(
            updates.len(),
            weights.len(),
            "Updates and weights must have the same length"
        );
        assert!(
            weights.iter().all(|&w| w >= 0.0),
            "Weights must be non-negative"
        );
        let weighted_index = WeightedIndex::new(&weights).unwrap_or_else(|_| {
            if weights.is_empty() {
                WeightedIndex::new(&[1.0]).unwrap()
            } else {
                panic!("Invalid weights");
            }
        });
        Self {
            system,
            action,
            updates,
            weights,
            rng,
            //step_count: 0,
            weighted_index,
        }
    }
    pub fn step(&mut self) -> Option<AcceptedUpdate> {
        if self.updates.is_empty() {
            return None;
        }
        let idx = self.weighted_index.sample(&mut self.rng);
        self.updates[idx].monte_carlo_update(&mut self.system, &self.action, &mut self.rng)
    }

    pub fn set_weight(&mut self, index: usize, weight: f64) {
        assert!(weight >= 0.0, "Weight must be non-negative");
        assert!(index < self.weights.len(), "Index out of bounds");
        self.weights[index] = weight;
        self.weighted_index =
            WeightedIndex::new(&self.weights).unwrap_or_else(|_| panic!("Invalid weights"));
    }

    pub fn push_update<U>(&mut self, update: U, weight: f64)
    where
        U: MonteCarloUpdate<S, A, R> + 'static,
    {
        assert!(weight >= 0.0, "Weight must be non-negative");
        self.updates.push(Box::new(update));
        self.weights.push(weight);
        self.weighted_index =
            WeightedIndex::new(&self.weights).unwrap_or_else(|_| panic!("Invalid weights"));
    }

    pub fn weights(&self) -> &[f64] {
        &self.weights
    }
}
