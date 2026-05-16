"""CLI: Download and save market return data.

Downloads US ETF returns from Yahoo Finance and MOEX index returns via apimoex,
saves each to a parquet file, and prints basic diagnostics.

Usage
-----
    python -m scripts.download_data
    python -m scripts.download_data --start 2010-01-01 --end 2024-12-31
    python -m scripts.download_data --skip-moex
    python -m scripts.download_data --log-level DEBUG
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import click

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data.loaders import download_yahoo_returns, download_moex_returns
from src.utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)

US_TICKERS: list[str] = [
    "SPY", "QQQ", "XLF", "XLE", "XLK",
    "XLV", "XLI", "XLY", "XLP", "TLT",
    "IEF", "GLD", "VNQ", "EEM", "EFA",
]

MOEX_TICKERS: list[str] = [
    "IMOEX", "MOEXOG", "MOEXFN", "MOEXMM", "MOEXTL", "RTSI",
]

DEFAULT_START = "2007-01-01"
DEFAULT_END = "2026-05-16"


def _print_stats(label: str, df) -> None:
    sep = "=" * 56
    click.echo(f"\n{sep}")
    click.echo(f"  {label}")
    click.echo(sep)
    click.echo(f"  Shape      : {df.shape[0]:,} rows x {df.shape[1]} cols")
    click.echo(
        f"  Date range : {df.index.min().date()} to {df.index.max().date()}"
    )
    click.echo("  % missing per ticker:")
    for ticker, pct in df.isnull().mean().mul(100).items():
        flag = "  !" if pct > 5 else "   "
        click.echo(f"{flag}  {ticker:<12}  {pct:6.2f}%")


@click.command()
@click.option("--start", default=DEFAULT_START, show_default=True, help="Start date (ISO).")
@click.option("--end", default=DEFAULT_END, show_default=True, help="End date (ISO).")
@click.option("--output-dir", "-o", default=None, help="Override data/raw output directory.")
@click.option("--skip-yahoo", is_flag=True, default=False, help="Skip Yahoo Finance download.")
@click.option("--skip-moex", is_flag=True, default=False, help="Skip MOEX download.")
@click.option("--log-level", default="INFO", show_default=True, help="Logging verbosity.")
def main(
    start: str,
    end: str,
    output_dir: str | None,
    skip_yahoo: bool,
    skip_moex: bool,
    log_level: str,
) -> None:
    """Download US ETF and MOEX index return data and save to parquet."""
    raw_dir = Path(output_dir) if output_dir else _ROOT / "data" / "raw"
    setup_logging(log_level)

    if not skip_yahoo:
        us_path = raw_dir / "us_etf_returns.parquet"
        try:
            df = download_yahoo_returns(US_TICKERS, start, end, save_path=us_path)
            _print_stats("US ETFs  (Yahoo Finance / auto_adjust=True)", df)
        except ImportError as exc:
            click.echo(f"[SKIP] Yahoo Finance: {exc}", err=True)

    if not skip_moex:
        moex_path = raw_dir / "moex_returns.parquet"
        try:
            df = download_moex_returns(MOEX_TICKERS, start, end, save_path=moex_path)
            if df is not None:
                _print_stats("MOEX Indices  (apimoex / ISS API)", df)
        except RuntimeError as exc:
            click.echo(f"[ERROR] MOEX download failed: {exc}", err=True)

    click.echo("\nDone.")


if __name__ == "__main__":
    main()
