import os
import argparse
import logging
from mediner import commands


logger = logging.getLogger(__name__)


def get_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(
		prog='mediner',
		description='Medical NER CLI Tool'
	)
	parser.add_argument('--convert-csv-to-label-studio')
	parser.add_argument('--text-column')
	parser.add_argument('--output-json')
	parser.add_argument('--debug', action='store_true')
	return parser


def main() -> None:
	"""Main CLI input for mediner

	:return None:
	"""
	parser = get_parser()
	args = parser.parse_args()
	if args.debug:
		logging.basicConfig(level=logging.DEBUG)
	else:
		logging.basicConfig(level=logging.INFO)

	if args.convert_csv_to_label_studio and args.text_column:
		commands.convert_csv_to_label_studio(args.convert_csv_to_label_studio, args.text_column, args.output_json)