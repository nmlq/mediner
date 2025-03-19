import argparse
import logging
from mediner import commands


logger = logging.getLogger(__name__)


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='mediner',
        description='Medical NER CLI Tool'
    )
    parser.add_argument(
        '--convert-csv-to-label-studio',
        help="Convert CSV to importable label-studio format"
    )
    parser.add_argument(
        '--convert-jsons-to-label-studio',
        nargs='+',
        help=(
            "Convert exported json files"
            " from label-studio format "
            "to reimport, for adding predictions."
        )
    )
    parser.add_argument(
        '--text-column',
        help="Text column to use on CSV conversions; Default 'ReportText'",
        type=str,
        default='ReportText',
    )
    parser.add_argument(
        '--output-json',
        help="Output json for conversions"
    )
    parser.add_argument(
        '--model-filename',
        help="Model filename to load for conversion predictions"
    )
    parser.add_argument(
        '--predict',
        action='store_true',
        help=(
            "Enable predictions on "
            "conversions; Works with "
            "any `--convert` arg"
        )
    )
    parser.add_argument(
        '--train',
        action='store_true',
        help="Train a mediner model; Requires '--exported-jsons'"
    )
    parser.add_argument(
        '--exported-jsons',
        nargs='+',
        help="Exported json files output from label-studio"
    )
    parser.add_argument(
        '--load',
        help="Test loading a mediner model file"
    )
    parser.add_argument(
        '--k',
        type=int,
        default=None,
        help="k for k-fold cross-validation"
    )
    parser.add_argument(
        '--hold-out-percentage',
        type=float,
        default=0.2,
        help="percentage for hold-fold validation"
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help="Turn on debug logging"
    )
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
        commands.convert_csv_to_label_studio(
            args.convert_csv_to_label_studio,
            args.text_column,
            args.output_json,
            args.predict,
            args.model_filename
        )
    elif args.convert_jsons_to_label_studio:
        commands.convert_jsons_to_label_studio(
            args.convert_jsons_to_label_studio,
            args.output_json,
            args.predict,
            args.model_filename
        )
    elif args.train and args.exported_jsons:
        commands.train(
            args.exported_jsons,
            output_filename=None,
            k=args.k,
            percentage=args.hold_out_percentage
        )
    elif args.load:
        commands.load(args.load)
    else:
        logger.info("Couldn't understand args")
        parser.print_help()
