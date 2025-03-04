## To Do

* WorldLines:
    * ~~particle method should be moved to WorldLineDimensions~~
    * const PARTICLES?
* Improve ProposedUpdate design
* Action:
    * Implement single particle propagator methods
* Consider using FxHashMap instead of HashMap
* ~~In MonteCarloUpdate: rename `try_update` to `advance` or `apply`~~
* Implement nearest neighbour table as library cfg feature
    * Adapt swap with table
* Define a ``prelude'' for common imports 
* Update README with:
  * [![Latest version](https://img.shields.io/crates/v/pimc.svg)](https://crates.io/crates/pimc)
  * [![API](https://docs.rs/pimc/badge.svg)](https://docs.rs/pimc)
  * Modify licensing to "`pimc` is distributed under the MIT license. See LICENSE for details."
* Modify .github/workflows/rust.yml Name to "main tests".

* In pimc::monte_carlo::traits::MonteCarloTunable
  * Fix asymmetry in return type: set_parameter should return a bool/Option/Result to signal if the modification was successful.

