use env_logger::Builder;
use log::info;
use pimc_rs::path_state::worm::Worm;
use pimc_rs::updates::monte_carlo_update::MonteCarloUpdate;
use pimc_rs::updates::redraw::Redraw;
use pimc_rs::updates::worm_translate::WormTranslate;

const N: usize = 3;
const M: usize = 8;
const D: usize = 2;

const MP1: usize = M + 1;

fn two_lambda_tau(_: &Worm<N, MP1, D>, _: usize) -> f64 {
    1.0
}

fn main() {
    // Programmatically set the logging level to TRACE
    Builder::new().filter_level(log::LevelFilter::Trace).init();

    info!("Starting Example");
    let mut path = Worm::<N, MP1, D>::new();
    path.set_preceding(0, None);
    path.set_following(0, Some(2));
    path.set_preceding(2, Some(0));
    path.set_following(2, None);
    info!("Head is {:?}", path.worm_head());
    info!("Tail is {:?}", path.worm_tail());
    info!("Sector is {:?}", path.sector());

    let mut mc_transl = WormTranslate::new((&[1.0]).to_vec());
    let mut rng = rand::thread_rng();

    let success = mc_transl.try_update(&mut path, |_, _| 0.5, &mut rng);
    if success {
        for part in 0..path.particles() {
            info!(
                "New path for p = {}\n{:?}",
                part,
                path.positions(part, 0, path.time_slices())
            );
        }
    }

    let mut mc_redraw = Redraw::new(M / 3, M - 1, two_lambda_tau);
    let success = mc_redraw.try_update(&mut path, |_, _| 0.5, &mut rng);
    if success {
        for part in 0..path.particles() {
            info!(
                "New path for p = {}\n{:?}",
                part,
                path.positions(part, 0, path.time_slices())
            );
        }
    }
}
