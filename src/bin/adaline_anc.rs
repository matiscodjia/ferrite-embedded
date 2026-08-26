//! Adaptive noise cancellation (ADALINE / LMS), online training on-device.
//!
//! `Linear<WINDOW, 1, WINDOW>` with no hidden layer, `mse`, `Sgd`: this
//! configuration alone reproduces the Widrow-Hoff LMS update rule (Widrow &
//! Hoff, 1960) exactly. Trained one sample at a time against a simulated
//! primary/reference microphone pair: a clean multi-tone signal mixed with
//! multi-harmonic (engine-like) noise, and a reference input correlated with
//! the noise through a 2-tap coupling path but not with the signal -- the
//! standard adaptive-noise-cancellation setup. Each `DATA` line is a CSV
//! record consumed by `scripts/plot_adaline_anc.py`:
//!     DATA,n,s_true,n0,d_n,y_n,e_n,err_vs_truth,cycles
#![no_std]
#![no_main]

use core::sync::atomic::{compiler_fence, Ordering};
use cortex_m_rt::entry;
use cortex_m_rt::{exception, ExceptionFrame};
use defmt_rtt as _;
use ferrite::autodiff::{mse, Linear, Module, Update};
use ferrite::{Scalar, Vector};
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

const WINDOW: usize = 16;
const N_SAMPLES: u32 = 3000;
const FS: f32 = 8000.0;
const F_SIGNAL: [f32; 3] = [60.0, 140.0, 310.0];
const F_NOISE: [f32; 3] = [200.0, 400.0, 600.0];
const A_SIGNAL: [f32; 3] = [0.3, 0.3, 0.3];
const A_NOISE: [f32; 3] = [0.5, 0.3, 0.2];
const LR: Scalar = 0.002;
// Unknown (to the ADALINE) 2-tap coupling from the underlying noise source
// to the reference microphone: x[n] = COUPLING[0]*w[n] + COUPLING[1]*w[n-1].
const COUPLING: [f32; 2] = [0.7, 0.3];

fn tone_sum(freqs: &[f32; 3], amps: &[f32; 3], n: u32) -> f32 {
    let t = n as f32 / FS;
    let mut acc = 0.0f32;
    let mut i = 0;
    while i < 3 {
        acc += amps[i] * libm::sinf(2.0 * core::f32::consts::PI * freqs[i] * t);
        i += 1;
    }
    acc
}

#[entry]
fn main() -> ! {
    let dp = pac::Peripherals::take().unwrap();
    let mut cp = cortex_m::Peripherals::take().unwrap();
    bench::init(&mut cp.DCB, &mut cp.DWT);

    // Sans ceci, DWT compte des cycles a 16 MHz (HSI par defaut) alors que
    // la conversion en us plus bas suppose 168 MHz (meme correctif que
    // conv.rs/compute.rs).
    let rcc = dp.RCC.constrain();
    let clocks = rcc.cfgr.use_hse(8.MHz()).sysclk(168.MHz()).freeze();
    defmt::info!("sysclk reel = {} Hz", clocks.sysclk().raw());
    defmt::info!(
        "ADALINE ANC : WINDOW={} N_SAMPLES={} fs={} Hz | f_signal=[{}, {}, {}] Hz f_bruit=[{}, {}, {}] Hz",
        WINDOW as u32,
        N_SAMPLES,
        FS,
        F_SIGNAL[0],
        F_SIGNAL[1],
        F_SIGNAL[2],
        F_NOISE[0],
        F_NOISE[1],
        F_NOISE[2]
    );
    let budget_cycles = (168_000_000.0 / FS) as u32;
    defmt::info!("budget cycles/echantillon a fs={}Hz : {}", FS, budget_cycles);

    let mut net = Linear::<WINDOW, 1, WINDOW>::zeros();

    // x_hist[0] = most recent reference sample, x_hist[WINDOW-1] = oldest.
    let mut x_hist = [0.0 as Scalar; WINDOW];
    let mut w_prev: f32 = 0.0;

    let mut sum_cycles: u64 = 0;
    let mut max_cycles: u32 = 0;

    let mut n: u32 = 0;
    while n < N_SAMPLES {
        let s = tone_sum(&F_SIGNAL, &A_SIGNAL, n);
        let w = tone_sum(&F_NOISE, &A_NOISE, n);
        let x = COUPLING[0] * w + COUPLING[1] * w_prev;
        w_prev = w;
        let n0 = w;
        let d = s + n0;

        let mut i = WINDOW - 1;
        while i > 0 {
            x_hist[i] = x_hist[i - 1];
            i -= 1;
        }
        x_hist[0] = x as Scalar;

        let input = Vector::<WINDOW>::from_data(x_hist);
        let target = Vector::<1>::from_data([d as Scalar]);

        compiler_fence(Ordering::SeqCst);
        let t0 = bench::cycles();
        let (output, ctx) = net.forward(input);
        let (_, grad_out) = mse(output, target);
        let (_, grads) = net.backward(grad_out, &ctx);
        net.update(&grads, LR);
        let elapsed = bench::cycles().wrapping_sub(t0);
        compiler_fence(Ordering::SeqCst);

        sum_cycles += elapsed as u64;
        if elapsed > max_cycles {
            max_cycles = elapsed;
        }

        if n % 3 == 0 {
            let y = output[0] as f32;
            let e = d - y;
            let err_vs_truth = e - s;
            defmt::info!(
                "DATA,{},{},{},{},{},{},{},{}",
                n,
                s,
                n0,
                d,
                y,
                e,
                err_vs_truth,
                elapsed
            );
        }

        n += 1;
    }

    let mean_cycles = sum_cycles as f32 / N_SAMPLES as f32;
    defmt::info!(
        "resume : cycles/echantillon moyen = {} ({} us) | pic = {} ({} us)",
        mean_cycles,
        mean_cycles / 168.0,
        max_cycles,
        max_cycles as f32 / 168.0
    );

    defmt::info!("=== fin adaline_anc ===");

    loop {}
}
