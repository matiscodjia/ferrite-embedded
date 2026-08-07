//! Calibration du débit USART2 réel (VCP ST-Link, /dev/ttyACM0 côté hôte) :
//! echo simple octet par octet. Le baud est un `const` à changer entre deux
//! essais — pas de négociation dynamique, le but est juste de mesurer le
//! plafond réel avant de concevoir le protocole vidéo dessus (voir
//! scripts/hil/bench_uart.py côté hôte).
#![no_std]
#![no_main]

use cortex_m_rt::entry;
use cortex_m_rt::{ExceptionFrame, exception};
use defmt_rtt as _;
use panic_probe as _;
use stm32f4xx_hal::block;
use stm32f4xx_hal::{pac, prelude::*, serial::config::Config};

#[exception]
unsafe fn HardFault(ef: &ExceptionFrame) -> ! {
    defmt::error!("HardFault ! PC={:x} SP={:x}", ef.pc(), ef.lr());
    loop {}
}

/// Candidat à tester — changer et reflasher entre deux mesures.
const BAUD: u32 = 921_600;

#[entry]
fn main() -> ! {
    let dp = pac::Peripherals::take().unwrap();
    let _cp = cortex_m::Peripherals::take().unwrap();

    let rcc = dp.RCC.constrain();
    let clocks = rcc.cfgr.use_hse(8.MHz()).sysclk(168.MHz()).freeze();

    let gpioa = dp.GPIOA.split();
    let tx_pin = gpioa.pa2.into_alternate();
    let rx_pin = gpioa.pa3.into_alternate();
    let mut serial = dp
        .USART2
        .serial(
            (tx_pin, rx_pin),
            Config::default().baudrate(BAUD.bps()),
            &clocks,
        )
        .unwrap();

    defmt::info!(
        "uart_bench pret : sysclk={} Hz, baud USART2={}",
        clocks.sysclk().raw(),
        BAUD
    );

    loop {
        let byte: u8 = block!(serial.read()).unwrap();
        block!(serial.write(byte)).unwrap();
    }
}
