//! A/B reel sur cible : Tensor::multiply (get/set, verifie) vs
//! Tensor::multiply_unchecked (get_unchecked/set_unchecked) -- meme boucle,
//! meme forme de matrice que l'ADALINE (1x16 * 16x1), seule la methode
//! d'acces change. black_box sur les entrees a chaque iteration (meme
//! patron que compute.rs/regime_bench.rs deja valide sur ce repo) pour
//! empecher LLVM de constant-propager tout le calcul hors de la boucle.
#![no_std]
#![no_main]

use core::hint::black_box;
use core::sync::atomic::{compiler_fence, Ordering};
use cortex_m_rt::entry;
use cortex_m_rt::{exception, ExceptionFrame};
use defmt_rtt as _;
use frugal_ml::linalg::tensor::{Tensor, Vector};
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

const ITERS: u32 = 2000;

#[entry]
fn main() -> ! {
    let dp = pac::Peripherals::take().unwrap();
    let mut cp = cortex_m::Peripherals::take().unwrap();
    bench::init(&mut cp.DCB, &mut cp.DWT);

    let rcc = dp.RCC.constrain();
    let clocks = rcc.cfgr.use_hse(8.MHz()).sysclk(168.MHz()).freeze();
    defmt::info!("sysclk reel = {} Hz", clocks.sysclk().raw());
    defmt::info!("A/B multiply vs multiply_unchecked, forme 1x16 * 16x1, {} iterations", ITERS);

    let w: Tensor<1, 16, 16> = Tensor::new([
        0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.9, -0.1, 0.2, -0.3, 0.4, -0.5, 0.6, -0.7,
    ]);
    let x: Vector<16> = Vector::from_data([
        1.0, 0.5, -0.5, 0.25, -0.25, 0.1, -0.1, 0.75, -0.75, 0.3, -0.3, 0.6, -0.6, 0.9, -0.9, 0.2,
    ]);

    let mut acc: f32 = 0.0;

    compiler_fence(Ordering::SeqCst);
    let t0 = bench::cycles();
    compiler_fence(Ordering::SeqCst);
    for _ in 0..ITERS {
        let r = black_box(&w).multiply::<1, 16, 1>(black_box(&x));
        acc += r.get(0, 0);
    }
    compiler_fence(Ordering::SeqCst);
    let t1 = bench::cycles();
    compiler_fence(Ordering::SeqCst);
    let checked_total = t1.wrapping_sub(t0);

    compiler_fence(Ordering::SeqCst);
    let t2 = bench::cycles();
    compiler_fence(Ordering::SeqCst);
    for _ in 0..ITERS {
        let r = black_box(&w).multiply_unchecked::<1, 16, 1>(black_box(&x));
        acc += r.get(0, 0);
    }
    compiler_fence(Ordering::SeqCst);
    let t3 = bench::cycles();
    compiler_fence(Ordering::SeqCst);
    let unchecked_total = t3.wrapping_sub(t2);

    defmt::info!(
        "CHECKED   : {} cycles total, {} cycles/iter ({} ns)",
        checked_total,
        checked_total / ITERS,
        (checked_total / ITERS) as f32 * 1000.0 / 168.0
    );
    defmt::info!(
        "UNCHECKED : {} cycles total, {} cycles/iter ({} ns)",
        unchecked_total,
        unchecked_total / ITERS,
        (unchecked_total / ITERS) as f32 * 1000.0 / 168.0
    );
    defmt::info!(
        "ratio checked/unchecked = {}",
        checked_total as f32 / unchecked_total as f32
    );
    defmt::info!("acc (anti-elimination) = {}", acc);

    loop {}
}
