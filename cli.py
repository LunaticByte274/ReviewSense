#!/usr/bin/env python3
"""
ReviewSense - Standalone Headless Batch Processing CLI
Enables automated batch NLP evaluation, CSV enrichment, and executive report
generation from terminal, CI pipelines, or cron workflows without starting a browser.
"""

import sys
import argparse
from pathlib import Path
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import AppConfig
from src.data_manager import DataManager, sanitize_dataframe_for_export
from src.analytics import AnalyticsEngine
from src.report_generator import ReportGenerator
from src.logger import logger


def main():
    parser = argparse.ArgumentParser(
        prog="reviewsense-cli",
        description=f"{AppConfig.APP_NAME} v{AppConfig.APP_VERSION} - Enterprise Headless Batch Review Intelligence"
    )
    parser.add_argument("-i", "--input", required=True, help="Path to input CSV file containing customer reviews.")
    parser.add_argument("-o", "--output", help="Path to output enriched CSV with sentiment & polarity scores.")
    parser.add_argument("-r", "--report", help="Path to generate standalone Executive HTML Intelligence Report.")
    parser.add_argument("-t", "--text-column", help="Explicit column header containing review text (optional; auto-detected if omitted).")
    parser.add_argument("--version", action="version", version=f"{AppConfig.APP_NAME} v{AppConfig.APP_VERSION}")

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.is_file():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)

    logger.info(f"Loading input dataset: {input_path}")
    with open(input_path, "rb") as f:
        file_bytes = f.read()

    df, err = DataManager.load_csv_bytes(file_bytes, filename=input_path.name)
    if err or df is None:
        logger.error(f"Failed to parse CSV: {err}")
        sys.exit(1)

    # Determine text column
    if args.text_column:
        text_col = args.text_column
        if text_col not in df.columns:
            logger.error(f"Specified column '{text_col}' not found in CSV. Available columns: {list(df.columns)}")
            sys.exit(1)
    else:
        detected = DataManager.detect_candidate_columns(df)
        text_col = detected.get("text_column")
        if not text_col or text_col not in df.columns:
            logger.error("Could not auto-detect review text column. Please specify with --text-column.")
            sys.exit(1)
        logger.info(f"Auto-detected review text column: '{text_col}'")

    logger.info(f"Executing batch NLP scoring over {len(df):,} records...")
    processed_df = DataManager.process_dataframe(df, text_column=text_col)

    # Calculate summary KPIs
    kpis = AnalyticsEngine.calculate_kpis(processed_df)
    nss = round(kpis["positive_pct"] - kpis["negative_pct"], 1)

    print("\n" + "=" * 55)
    print(f"⚡ {AppConfig.APP_NAME} - Batch Analysis Summary")
    print("=" * 55)
    print(f"Total Reviews Scored : {kpis['total_reviews']:,} ({kpis['valid_reviews']:,} valid)")
    print(f"Positive Sentiment   : {kpis['positive_count']:,} ({kpis['positive_pct']}%)")
    print(f"Negative Sentiment   : {kpis['negative_count']:,} ({kpis['negative_pct']}%)")
    print(f"Neutral Sentiment    : {kpis['neutral_count']:,} ({kpis['neutral_pct']}%)")
    print(f"Mean Polarity        : {kpis['avg_polarity']:+.3f}")
    print(f"Mean Subjectivity    : {kpis['avg_subjectivity']:.3f}")
    print(f"Net Sentiment (NSS)  : {nss:+0.1f}")
    print("=" * 55 + "\n")

    # Export CSV if requested
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        safe_export = sanitize_dataframe_for_export(processed_df)
        safe_export.to_csv(out_path, index=False)
        logger.info(f"Saved enriched CSV with injection defense: {out_path}")

    # Export Executive HTML Report if requested
    if args.report:
        rep_path = Path(args.report)
        rep_path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "source": f"CLI Ingestion: {input_path.name}",
            "filename": input_path.name,
            "total_rows": len(processed_df),
            "text_column": text_col
        }
        html_content = ReportGenerator.generate_html_report(processed_df, metadata, text_col=text_col)
        with open(rep_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        logger.info(f"Saved Executive HTML Report: {rep_path}")

    logger.info("Batch execution completed successfully.")


if __name__ == "__main__":
    main()
