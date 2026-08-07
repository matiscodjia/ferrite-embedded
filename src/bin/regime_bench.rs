//! Passe en revue plusieurs régimes (résolution, canaux, nombre de filtres)
//! sur `cross_correlate2d`/`tensordot_3`, mesure les cycles réels via DWT et
//! calcule l'analyse (MACs, RAM, %tick à 10ms/100Hz) directement sur cible —
//! pas de post-traitement nécessaire pour lire le verdict en RTT, mais chaque
//! régime finit aussi par une ligne `REGIME|...` compacte pour un parsing
//! automatique côté host (voir scripts/).
#![no_std]
#![no_main]

use core::hint::black_box;
use cortex_m_rt::entry;
use cortex_m_rt::{ExceptionFrame, exception};
use defmt_rtt as _;
use ferrite::linalg::{Tensor3D, Tensor4D, tensordot_3};
use ferrite::sp::{Gaussian3D, filter_bank};
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

/// Budget RAM de sécurité pour les tenseurs d'un régime (entrée + banc de
/// filtres + sortie) : 100 Ko sur les 128 Ko totaux, le reste couvrant la
/// pile d'appel, les ISR et le reste du programme. Dépassé -> régime marqué
/// SKIP, jamais alloué (éviter un stack overflow qui interromprait la suite
/// des mesures).
const RAM_SAFETY_BUDGET: usize = 100_000;
/// Fréquence coeur réelle après `sysclk(168.MHz())` plus bas.
const SYSCLK_HZ: f32 = 168_000_000.0;
const TICK_BUDGET_US: f32 = 10_000.0; // 10ms, cadence cible 100Hz

macro_rules! bench_regime {
    ($name:literal, H=$h:expr, W=$w:expr, C=$c:expr, K=$k:expr, KH=$kh:expr, KW=$kw:expr, stride=$s:expr, iters=$iters:expr) => {{
        // #[inline(never)] force un vrai appel de fonction : sa pile est
        // reprise à `ret`, avant l'appel du régime suivant. Sans ça, `main`
        // ne retournant jamais, les 6 régimes (types de tenseurs distincts,
        // donc aucune réutilisation d'emplacement pile garantie par LLVM)
        // additionnent leurs tailles dans un seul prologue au lieu de se
        // libérer entre deux — c'est ce qui a fait exploser la pile plus tôt
        // (gray96_k1 + gray64_k1 + ... ≈ 265 Ko pour 128 Ko de RAM).
        #[inline(never)]
        fn run_regime() {
        ferrite::conv_shape!(regime, N = 1, C = $c, H = $h, W = $w, K = $k, KH = $kh, KW = $kw, stride = $s);
        const RAM_BYTES: usize = 4 * (regime::NUMEL_X + regime::NUMEL_F + regime::NUMEL_Y);
        const MACS: usize =
            regime::H_OUT * regime::W_OUT * regime::C * regime::KH * regime::KW * regime::K;

        defmt::info!(
            "{} | H={} W={} C={} K={} noyau={}x{} | sortie {}x{} | MACs={} | tenseurs={} o ({}% des 128Ko)",
            $name,
            regime::H,
            regime::W,
            regime::C,
            regime::K,
            regime::KH,
            regime::KW,
            regime::H_OUT,
            regime::W_OUT,
            MACS,
            RAM_BYTES,
            (RAM_BYTES as f32 / 131072.0) * 100.0
        );

        if RAM_BYTES > RAM_SAFETY_BUDGET {
            defmt::warn!(
                "  -> SKIP : {} o dépasse le budget RAM de sécurité ({} o) - non exécuté pour ne pas faire déborder la pile",
                RAM_BYTES,
                RAM_SAFETY_BUDGET
            );
            defmt::info!(
                "REGIME|{}|{}|{}|{}|{}|{}|{}|{}|{}|SKIP|0|0.0|0.0|0.0",
                $name, regime::H, regime::W, regime::C, regime::K, regime::KH, regime::KW, MACS, RAM_BYTES
            );
        } else {
            const NUMEL_KERNEL: usize = $c * $kh * $kw;
            let frame = Tensor4D::<
                { regime::N },
                { regime::C },
                { regime::H },
                { regime::W },
                { regime::NUMEL_X },
            >::new([1.0; regime::NUMEL_X]);
            let kernel: Tensor3D<{ $c }, { $kh }, { $kw }, { NUMEL_KERNEL }> =
                Gaussian3D::kernel();
            let filters: Tensor4D<
                { regime::K },
                { regime::C },
                { regime::KH },
                { regime::KW },
                { regime::NUMEL_F },
            > = filter_bank([&kernel; regime::K]);

            let t0 = bench::cycles();
            for _ in 0..$iters {
                let out: Tensor4D<
                    { regime::N },
                    { regime::H_OUT },
                    { regime::W_OUT },
                    { regime::K },
                    { regime::NUMEL_Y },
                > = tensordot_3(
                    black_box(
                        &frame
                            .im2col_view::<{ regime::H_OUT }, { regime::W_OUT }, { regime::KH }, { regime::KW }>(
                                regime::STRIDE,
                            ),
                    ),
                    black_box(&filters),
                );
                black_box(out);
            }
            let elapsed = bench::cycles().wrapping_sub(t0);
            let per_iter = elapsed / ($iters as u32);
            let cycles_per_mac = per_iter as f32 / MACS as f32;
            let time_us = per_iter as f32 / (SYSCLK_HZ / 1_000_000.0);
            let pct_tick = (time_us / TICK_BUDGET_US) * 100.0;

            defmt::info!(
                "  -> {} cycles/iter ({} us, {} cycles/MAC) | {}% du tick 10ms",
                per_iter,
                time_us,
                cycles_per_mac,
                pct_tick
            );
            if pct_tick <= 50.0 {
                defmt::info!("  -> OK : tient dans la moitié du tick, marge confortable");
            } else if pct_tick <= 100.0 {
                defmt::warn!("  -> SERRE : tient dans le tick mais laisse peu de marge pour le reste");
            } else {
                defmt::error!("  -> DEPASSE le tick de 10ms a lui seul");
            }

            defmt::info!(
                "REGIME|{}|{}|{}|{}|{}|{}|{}|{}|{}|OK|{}|{}|{}|{}",
                $name,
                regime::H,
                regime::W,
                regime::C,
                regime::K,
                regime::KH,
                regime::KW,
                MACS,
                RAM_BYTES,
                per_iter,
                cycles_per_mac,
                time_us,
                pct_tick
            );
        }
        }
        run_regime();
    }};
}

#[entry]
fn main() -> ! {
    let dp = pac::Peripherals::take().unwrap();
    let mut cp = cortex_m::Peripherals::take().unwrap();
    bench::init(&mut cp.DCB, &mut cp.DWT);

    let rcc = dp.RCC.constrain();
    let clocks = rcc.cfgr.use_hse(8.MHz()).sysclk(168.MHz()).freeze();
    defmt::info!("sysclk reel = {} Hz", clocks.sysclk().raw());
    defmt::info!(
        "=== regime_bench: budget tick=10ms (100Hz), budget RAM securite={} o / 128 Ko ===",
        RAM_SAFETY_BUDGET
    );

    // Le régime de départ (celui déjà mesuré à la main) sert de témoin pour
    // vérifier que la correction d'horloge ne change rien aux MACs/cycle.
    bench_regime!(
        "gray96_k1",
        H = 96,
        W = 96,
        C = 1,
        K = 1,
        KH = 3,
        KW = 3,
        stride = 1,
        iters = 10
    );
    bench_regime!(
        "gray64_k1",
        H = 64,
        W = 64,
        C = 1,
        K = 1,
        KH = 3,
        KW = 3,
        stride = 1,
        iters = 10
    );
    bench_regime!(
        "gray64_k4",
        H = 64,
        W = 64,
        C = 1,
        K = 4,
        KH = 3,
        KW = 3,
        stride = 1,
        iters = 5
    );
    bench_regime!(
        "rgb48_k4",
        H = 48,
        W = 48,
        C = 3,
        K = 4,
        KH = 3,
        KW = 3,
        stride = 1,
        iters = 5
    );
    bench_regime!(
        "rgb32_k4",
        H = 32,
        W = 32,
        C = 3,
        K = 4,
        KH = 3,
        KW = 3,
        stride = 1,
        iters = 5
    );
    // Volontairement hors du budget RAM (~129 Ko de tenseurs pour 128 Ko de
    // RAM totale) : doit ressortir SKIP, jamais alloué, jamais essayé.
    bench_regime!(
        "gray128_k1",
        H = 128,
        W = 128,
        C = 1,
        K = 1,
        KH = 3,
        KW = 3,
        stride = 1,
        iters = 5
    );

    defmt::info!("=== fin regime_bench ===");
    loop {}
}
