//! Mesure le coût réel (cycles DWT) d'une ligne dans `ConvStreaming`
//! (`push_row` + `conv2d`, le pipeline complet par ligne) pour plusieurs
//! largeurs de capteur, puis en déduit le plafond de FPS compute-bound pour
//! quelques hauteurs de frame usuelles — la RAM de `ConvStreaming` est
//! `O(KH * W)` (un ring buffer de KH lignes, jamais la frame entière), donc
//! contrairement à `regime_bench` il n'y a pas de garde-fou RAM à faire ici :
//! le point de cette structure est justement que la largeur ne coûte presque
//! rien en mémoire, seulement en cycles.
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
/// Hauteurs de frame usuelles pour lesquelles on dérive le FPS plafond :
/// QQVGA/QVGA/480p/720p.
const TARGET_HEIGHTS: [(&str, u32); 4] =
    [("120p", 120), ("240p", 240), ("480p", 480), ("720p", 720)];

macro_rules! bench_stream {
    ($name:literal, W=$w:expr, KH=$kh:expr, KW=$kw:expr, iters=$iters:expr) => {{
        const W: usize = $w;
        const KH: usize = $kh;
        const KW: usize = $kw;
        const W_OUT: usize = W - KW + 1;
        const MACS_PER_ROW: usize = W_OUT * KH * KW;
        const RAM_BYTES: usize = KH * W * core::mem::size_of::<Scalar>();

        defmt::info!(
            "{} | W={} noyau={}x{} | sortie {} px/ligne | MACs/ligne={} | ring buffer={} o",
            $name,
            W,
            KH,
            KW,
            W_OUT,
            MACS_PER_ROW,
            RAM_BYTES
        );

        let mut cs = ConvStreaming::<W, KH, KW>::new();
        let kernel: [[Scalar; KW]; KH] = [[1.0; KW]; KH];
        let row: [Scalar; W] = [1.0; W];

        // Amorce le ring buffer (KH lignes) hors mesure : c'est un régime
        // transitoire de démarrage, pas le coût par ligne en régime établi.
        for _ in 0..KH {
            cs.push_row(black_box(row));
        }

        let t0 = bench::cycles();
        for _ in 0..$iters {
            cs.push_row(black_box(row));
            let out: [Scalar; W_OUT] = cs.conv2d(black_box(&kernel));
            black_box(out);
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
            let max_fps = SYSCLK_HZ / (h as f32 * per_row as f32);
            defmt::info!("       {} ({} lignes) -> {} fps max", label, h, max_fps);
        }

        let fps_120 = SYSCLK_HZ / (120.0 * per_row as f32);
        let fps_240 = SYSCLK_HZ / (240.0 * per_row as f32);
        let fps_480 = SYSCLK_HZ / (480.0 * per_row as f32);
        let fps_720 = SYSCLK_HZ / (720.0 * per_row as f32);
        defmt::info!(
            "STREAM|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}",
            $name,
            W,
            KH,
            KW,
            per_row,
            us_per_row,
            cycles_per_mac,
            fps_120,
            fps_240,
            fps_480,
            fps_720
        );
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
    defmt::info!("=== stream_bench: ConvStreaming, cout par ligne + FPS plafond derive ===");

    bench_stream!("w160_k3", W = 160, KH = 3, KW = 3, iters = 50);
    bench_stream!("w320_k3", W = 320, KH = 3, KW = 3, iters = 50);
    bench_stream!("w640_k3", W = 640, KH = 3, KW = 3, iters = 30);
    bench_stream!("w1280_k3", W = 1280, KH = 3, KW = 3, iters = 20);
    // Même largeur (320), noyau plus gros : isole le coût du noyau de celui
    // de la largeur pour les deux séries w320_k3 / w320_k5.
    bench_stream!("w320_k5", W = 320, KH = 5, KW = 5, iters = 50);

    defmt::info!("=== fin stream_bench ===");
    loop {}
}
