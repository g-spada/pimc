use env_logger::Builder;
use log::info;
use ndarray::{Array1, ArrayView1};
use pimc_rs::path_state::worm::Worm;
use pimc_rs::space::free_space2::FreeSpace;
use pimc_rs::updates::monte_carlo_update::MonteCarloUpdate;
use pimc_rs::updates::open_close::OpenClose;
use pimc_rs::updates::proposed_update::ProposedUpdate;
use pimc_rs::updates::redraw::Redraw;
use pimc_rs::updates::redraw_head::RedrawHead;
use pimc_rs::updates::redraw_tail::RedrawTail;
use pimc_rs::updates::swap::Swap;
use pimc_rs::updates::worm_translate::WormTranslate;

const N: usize = 3;
const M: usize = 8;
const D: usize = 2;

const MP1: usize = M + 1;

fn two_lambda_tau(_: &Worm<N, MP1, D>, _: usize) -> f64 {
    0.1
}

fn distance(r1: ArrayView1<f64>, r2: ArrayView1<f64>) -> Array1<f64> {
    r1.to_owned() - r2
}

fn main() {
    // Programmatically set the logging level to TRACE
    Builder::new().filter_level(log::LevelFilter::Trace).init();

    info!("Starting Example");
    let mut path = Worm::<N, MP1, D>::new();
    // Create a worm of length two: (T) - 0 - 2 - (H)
    path.set_preceding(0, None);
    path.set_following(0, Some(2));
    path.set_preceding(2, Some(0));
    path.set_following(2, None);
    
    // Print starting configuration
    info!("Starting configuration:\n{:?}", path);

    // RNG
    let mut rng = rand::thread_rng();

    // Translate update
    let mut mc_transl = WormTranslate::new([1.0, 2.0], |_, _| 0.5);

    // Redraw update
    let mut mc_redraw = Redraw::new(M / 3, M - 1, two_lambda_tau, |_, _| 0.5);

    // RedrawHead update
    let mut mc_redrawhead = RedrawHead::new(M / 3, M - 1, two_lambda_tau, |_, _| 0.5);

    // RedrawTail update
    let mut mc_redrawtail = RedrawTail::new(M / 3, M - 1, two_lambda_tau, |_, _| 0.5);

    // Open/Close update
    let mut mc_openclose = OpenClose::new(
        M / 3,
        M - 1,
        0.5,
        1.0,
        4.0,
        distance,
        two_lambda_tau,
        |_, _| 0.5,
    );

    // Space object
    let flatlandia = FreeSpace::<D>;

    // Swap update
    let mut mc_swap = Swap::new(M / 3, M - 1, two_lambda_tau, |_, _| 0.5, &flatlandia);

    for mc_it in 0..10 {
        info!("######################################");
        info!("# ITERATION {}", mc_it);

        let success = mc_transl.try_update(&mut path, &mut rng);
        if let Some(update) = success {
            for part in update.modified_particles {
                info!(
                    "New path for p = {}\n{:?}",
                    part,
                    path.positions(part, 0, path.time_slices())
                );
            }
        }

        let success = mc_redraw.try_update(&mut path, &mut rng);
        if let Some(update) = success {
            for part in update.modified_particles {
                info!(
                    "New path for p = {}\n{:?}",
                    part,
                    path.positions(part, 0, path.time_slices())
                );
            }
        }

        let success = mc_redrawhead.try_update(&mut path, &mut rng);
        if let Some(update) = success {
            for part in update.modified_particles {
                info!(
                    "New path for p = {}\n{:?}",
                    part,
                    path.positions(part, 0, path.time_slices())
                );
            }
        }

        let success = mc_redrawtail.try_update(&mut path, &mut rng);
        if let Some(update) = success {
            for part in update.modified_particles {
                info!(
                    "New path for p = {}\n{:?}",
                    part,
                    path.positions(part, 0, path.time_slices())
                );
            }
        }

        let success = mc_openclose.try_update(&mut path, &mut rng);
        if let Some(update) = success {
            for part in update.modified_particles {
                info!(
                    "New path for p = {}\n{:?}",
                    part,
                    path.positions(part, 0, path.time_slices())
                );
            }
        }

        let success = mc_swap.try_update(&mut path, &mut rng);
        if let Some(update) = success {
            for part in update.modified_particles {
                info!(
                    "New path for p = {}\n{:?}",
                    part,
                    path.positions(part, 0, path.time_slices())
                );
            }
        }

        let flatlandia = FreeSpace::<2>;
        // Print configuration
        info!("Current configuration:\n{:?}", path);
        info!("Head is {:?}", path.worm_head());
        info!("Tail is {:?}", path.worm_tail());
        info!("Sector is {:?}", path.sector());

        // Check permutations
        for particle in 0..N {
            if let Some(next) = path.following(particle) {
                let diff = path.position(particle, M).to_owned() - path.position(next, 0);
                assert!(
                    diff.iter().any(|&l| l.abs() < 1e-10),
                    "Particles {} and {} not correctly glued together: difference is {:}",
                    particle,
                    next,
                    diff
                )
            }
        }
    }
}
