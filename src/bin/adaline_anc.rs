//! Banc d'essai ADALINE / LMS pour annulation adaptative de bruit (ANC),
//! sur cible reelle. Aucune nouvelle brique frugal_ml : Linear<WINDOW,1,NUMEL>
//! (pas d'activation) + mse + Sgd + Module::forward/backward directement --
//! c'est exactement le LMS de Widrow, derive formule par formule dans la
//! session qui precede ce fichier.
//!
//! Signal simule (schema ANC classique) :
//!   s[n]  = signal utile, sinusoide basse frequence (a preserver)
//!   n0[n] = bruit source, sinusoide + jitter (a annuler)
//!   x[n]  = entree de reference, n0[n] passe par un chemin de couplage
//!           INCONNU (filtre a 2 coefficients) -- correle au bruit, pas
//!           au signal utile
//!   d[n]  = s[n] + n0[n], l'entree primaire (ce que capterait un micro)
//!   y[n]  = sortie de l'ADALINE, estimation adaptative du bruit
//!   e[n]  = d[n] - y[n], signal annule -- doit converger vers s[n]
//!
//! err_vs_truth = e[n] - s[n] n'est mesurable qu'en simulation (on connait
//! le signal propre) -- sert de verite terrain pour juger la convergence.
#![no_std]
#![no_main]

use core::f32::consts::PI;
use core::sync::atomic::{compiler_fence, Ordering};
use cortex_m_rt::entry;
use cortex_m_rt::{exception, ExceptionFrame};
use defmt_rtt as _;
use frugal_ml::autodiff::core::module::Module;
use frugal_ml::autodiff::layers::linear::Linear;
use frugal_ml::autodiff::loss::mse;
use frugal_ml::autodiff::optim::optimizer::Optimizer;
use frugal_ml::autodiff::optim::sgd::Sgd;
use frugal_ml::linalg::tensor::Vector;
use frugal_ml::Scalar;
use libm::sinf;
use panic_probe as _;
use stm32f4xx_hal::{pac, prelude::*};

mod bench {
    use cortex_m::peripheral::{DCB, DWT};

    pub fn init(dcb: &mut DCB, dwt: &mut DWT) {
        dcb.enable_trace();
        dwt.enable_cycle_counter();
    }

    #[inline(always)]
    pub fn cycles() -> u32 {
        DWT::cycle_count()
    }
}

#[exception]
unsafe fn HardFault(ef: &ExceptionFrame) -> ! {
    defmt::error!("HardFault ! PC={:x} SP={:x}", ef.pc(), ef.lr());
    loop {}
}

fn xorshift_scalar(state: &mut u64) -> Scalar {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    (x as Scalar) / (u64::MAX as Scalar) * 2.0 - 1.0
}

const WINDOW: usize = 16;
const N_SAMPLES: u32 = 3000;
const FS: f32 = 8000.0;
// Signal utile = accord de 3 tons (pas une sinusoide pure) -- une "musique"
// simplifiee, frequences sans rapport harmonique avec le bruit.
const F_S: [f32; 3] = [60.0, 140.0, 310.0];
const A_S: [f32; 3] = [0.30, 0.15, 0.10];
// Bruit = fondamentale + harmoniques (comme un vrai bruit moteur), amplitude
// decroissante par harmonique, plus un peu de bruit large bande.
const F_N: [f32; 3] = [200.0, 400.0, 600.0];
const A_N: [f32; 3] = [0.50, 0.25, 0.15];

#[entry]
fn main() -> ! {
    let dp = pac::Peripherals::take().unwrap();
    let mut cp = cortex_m::Peripherals::take().unwrap();
    bench::init(&mut cp.DCB, &mut cp.DWT);

    let rcc = dp.RCC.constrain();
    let clocks = rcc.cfgr.use_hse(8.MHz()).sysclk(168.MHz()).freeze();
    defmt::info!("sysclk reel = {} Hz", clocks.sysclk().raw());
    defmt::info!(
        "ADALINE ANC : WINDOW={} N_SAMPLES={} fs={} Hz | f_signal={} Hz f_bruit={} Hz",
        WINDOW,
        N_SAMPLES,
        FS,
        F_S,
        F_N,
    );
    defmt::info!("budget cycles/echantillon a fs={}Hz : {}", FS, (168_000_000.0 / FS) as u32);

    let mut adaline = Linear::<WINDOW, 1, WINDOW>::zeros();
    let mut opt = Sgd::new(0.004);

    let mut window = [0.0 as Scalar; WINDOW];
    let mut rng_state: u64 = 424242;
    let mut n0_prev: Scalar = 0.0;

    let mut cycles_sum: u64 = 0;
    let mut cycles_max: u32 = 0;

    for n in 0..N_SAMPLES {
        let t = n as f32 / FS;
        let mut s_true: Scalar = 0.0;
        for k in 0..3 {
            s_true += A_S[k] * sinf(2.0 * PI * F_S[k] * t);
        }
        let mut n0: Scalar = 0.0;
        for k in 0..3 {
            n0 += A_N[k] * sinf(2.0 * PI * F_N[k] * t);
        }
        n0 += 0.03 * xorshift_scalar(&mut rng_state);
        // chemin de couplage inconnu : x[n] = 0.7*n0[n] + 0.3*n0[n-1]
        let x_n: Scalar = 0.7 * n0 + 0.3 * n0_prev;
        n0_prev = n0;
        let d_n: Scalar = s_true + n0;

        for i in (1..WINDOW).rev() {
            window[i] = window[i - 1];
        }
        window[0] = x_n;

        let input = Vector::from_data(window);
        let target = Vector::from_data([d_n]);

        compiler_fence(Ordering::SeqCst);
        let t0 = bench::cycles();
        compiler_fence(Ordering::SeqCst);
        let (y_vec, ctx) = adaline.forward(input);
        let (_loss, loss_grad) = mse(y_vec, target);
        let (_, grads) = adaline.backward(loss_grad, &ctx);
        opt.step(&mut adaline, &grads);
        compiler_fence(Ordering::SeqCst);
        let elapsed = bench::cycles().wrapping_sub(t0);
        compiler_fence(Ordering::SeqCst);

        cycles_sum += elapsed as u64;
        if elapsed > cycles_max {
            cycles_max = elapsed;
        }

        let y_n = y_vec[0];
        let e_n = d_n - y_n;
        let err_vs_truth = e_n - s_true;

        if n % 3 == 0 {
            defmt::info!(
                "DATA,{},{},{},{},{},{},{},{}",
                n,
                s_true,
                n0,
                d_n,
                y_n,
                e_n,
                err_vs_truth,
                elapsed
            );
        }
    }

    defmt::info!(
        "resume : cycles/echantillon moyen = {} ({} us) | pic = {} ({} us)",
        (cycles_sum / N_SAMPLES as u64) as u32,
        (cycles_sum as f32 / N_SAMPLES as f32) / 168.0,
        cycles_max,
        cycles_max as f32 / 168.0
    );

    loop {}
}
