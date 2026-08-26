//! Balaie deux axes indépendamment de la hauteur H (le point même de
//! `ConvStreaming` : sa RAM est `O(KH * W)`, jamais `O(H * W)`) :
//!
//! - **axe résolution** : W croissant à C=1 (GRIS) fixe ;
//! - **axe canaux** : C ∈ {1,2,3,4} à W fixe, répété à 4 largeurs (160,
//!   320, 640, 1280) pour vérifier que la pente (coût par canal ajouté)
//!   ne dépend pas de la résolution. C=1/C=3 sont les deux régimes réels
//!   (GRIS/RGB) ; C=2/C=4 sont des points de contrôle synthétiques qui ne
//!   correspondent à aucun capteur réel, ajoutés seulement pour confirmer
//!   la linéarité par régression plutôt que sur 2 points.
//!
//! Pour C canaux, on instancie C `ConvStreaming<W,3,3>` indépendants (un
//! ring buffer par canal) et on somme leurs sorties — exactement la
//! décomposition d'une conv multi-canaux (depthwise puis somme). Aucune
//! modification du cœur `ferrite` : composition de l'API publique
//! existante, même pattern que `stream_bench.rs`.
//!
//! Émet une ligne `SCALE|...` par régime pour parsing automatique côté
//! host (voir scripts/analyze.py).
#![no_std]
#![no_main]

use core::hint::black_box;
use cortex_m_rt::entry;
use cortex_m_rt::{ExceptionFrame, exception};
use defmt_rtt as _;
use ferrite::Scalar;
use ferrite::sp::ConvStreaming;
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

const SYSCLK_HZ: f32 = 168_000_000.0;
const TARGET_HEIGHTS: [(&str, u32); 4] =
    [("120p", 120), ("240p", 240), ("480p", 480), ("720p", 720)];

macro_rules! bench_scale {
    ($name:literal, C=$c:expr, W=$w:expr, iters=$iters:expr) => {{
        // #[inline(never)] : même raison que dans regime_bench.rs / stream_bench.rs
        // -- force la reprise de la pile de ce régime à `ret`, avant le suivant.
        #[inline(never)]
        fn run_scale() {
            const C: usize = $c;
            const W: usize = $w;
            const KH: usize = 3;
            const KW: usize = 3;
            const W_OUT: usize = W - KW + 1;
            const MACS_PER_ROW: usize = W_OUT * KH * KW * C;
            // RAM réellement proportionnelle à C (pente 4*KH*W, ordonnée à
            // l'origine nulle) -- contrairement au chemin plein-cadre de
            // regime_bench.rs, où le tenseur de sortie ajoute un terme
            // constant indépendant de C.
            const RAM_BYTES: usize = C * KH * W * core::mem::size_of::<Scalar>();

            defmt::info!(
                "{} | C={} W={} noyau={}x{} | sortie {} px/ligne | MACs/ligne={} | ring buffers={} o",
                $name, C, W, KH, KW, W_OUT, MACS_PER_ROW, RAM_BYTES
            );

            let mut channels: [ConvStreaming<W, KH, KW>; C] =
                core::array::from_fn(|_| ConvStreaming::new());
            let kernel: [[Scalar; KW]; KH] = [[1.0; KW]; KH];
            let row: [Scalar; W] = [1.0; W];

            // Amorce chaque ring buffer (KH lignes), hors mesure -- régime
            // transitoire de démarrage, pas le coût par ligne en régime établi.
            for _ in 0..KH {
                for ch in 0..C {
                    channels[ch].push_row(black_box(row));
                }
            }

            let t0 = bench::cycles();
            for _ in 0..$iters {
                let mut acc = [0.0; W_OUT];
                for ch in 0..C {
                    channels[ch].push_row(black_box(row));
                    // Pas de black_box() sur la sortie -- voir regime_bench.rs :
                    // ça a fait planter la carte en empêchant la réutilisation
                    // du même emplacement pile à chaque tour.
                    let out: [Scalar; W_OUT] = channels[ch].conv2d(black_box(&kernel));
                    for i in 0..W_OUT {
                        acc[i] += out[i];
                    }
                }
                let _ = acc;
            }
            let elapsed = bench::cycles().wrapping_sub(t0);
            let per_row = elapsed / ($iters as u32);
            let cycles_per_mac = per_row as f32 / MACS_PER_ROW as f32;
            let us_per_row = per_row as f32 / (SYSCLK_HZ / 1_000_000.0);

            defmt::info!(
                "  -> {} cycles/ligne ({} us, {} cycles/MAC)",
                per_row,
                us_per_row,
                cycles_per_mac
            );

            defmt::info!("  -> FPS plafond compute-bound (ignore acquisition capteur/DMA) :");
            for (label, h) in TARGET_HEIGHTS {
                let fps = SYSCLK_HZ / (h as f32 * per_row as f32);
                defmt::info!("       {} ({} lignes) -> {} fps max", label, h, fps);
            }

            let fps_120 = SYSCLK_HZ / (120.0 * per_row as f32);
            let fps_240 = SYSCLK_HZ / (240.0 * per_row as f32);
            let fps_480 = SYSCLK_HZ / (480.0 * per_row as f32);
            let fps_720 = SYSCLK_HZ / (720.0 * per_row as f32);
            defmt::info!(
                "SCALE|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}",
                $name,
                C,
                W,
                per_row,
                us_per_row,
                cycles_per_mac,
                RAM_BYTES,
                fps_120,
                fps_240,
                fps_480,
                fps_720
            );
        }
        run_scale();
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
        "=== scaling_bench: axe canaux (C=1..4) x axe resolution (W), ConvStreaming ==="
    );

    // Axe résolution : W croissant, C=1 (GRIS) -- la référence.
    bench_scale!("c1_w80", C = 1, W = 80, iters = 30);
    bench_scale!("c1_w160", C = 1, W = 160, iters = 30);
    bench_scale!("c1_w240", C = 1, W = 240, iters = 30);
    bench_scale!("c1_w320", C = 1, W = 320, iters = 30);
    bench_scale!("c1_w400", C = 1, W = 400, iters = 30);
    bench_scale!("c1_w480", C = 1, W = 480, iters = 30);
    bench_scale!("c1_w640", C = 1, W = 640, iters = 30);
    bench_scale!("c1_w800", C = 1, W = 800, iters = 30);
    bench_scale!("c1_w960", C = 1, W = 960, iters = 30);
    bench_scale!("c1_w1280", C = 1, W = 1280, iters = 20);

    // Axe canaux : C croissant à résolution fixée, répété à 4 largeurs pour
    // vérifier que la pente (coût par canal) ne dépend pas de la résolution.
    bench_scale!("c2_w160", C = 2, W = 160, iters = 30);
    bench_scale!("c3_w160", C = 3, W = 160, iters = 30);
    bench_scale!("c4_w160", C = 4, W = 160, iters = 30);

    bench_scale!("c2_w320", C = 2, W = 320, iters = 30);
    bench_scale!("c3_w320", C = 3, W = 320, iters = 30);
    bench_scale!("c4_w320", C = 4, W = 320, iters = 30);

    bench_scale!("c2_w640", C = 2, W = 640, iters = 30);
    bench_scale!("c3_w640", C = 3, W = 640, iters = 30);
    bench_scale!("c4_w640", C = 4, W = 640, iters = 30);

    bench_scale!("c2_w1280", C = 2, W = 1280, iters = 20);
    bench_scale!("c3_w1280", C = 3, W = 1280, iters = 20);
    bench_scale!("c4_w1280", C = 4, W = 1280, iters = 20);

    defmt::info!("=== fin scaling_bench ===");
    loop {}
}
