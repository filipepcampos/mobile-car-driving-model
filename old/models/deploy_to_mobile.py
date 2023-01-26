import argparse

import torch
import torch.utils.mobile_optimizer as mobile_optimizer
from loguru import logger

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Convert .pth models to mobile .ptl models.",
    )

    parser.add_argument(
        "-m",
        "--model",
        type=str,
        help="input model",
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=False,
        help="base name of the output files",
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    logger.info("Loading the model")
    try:
        model = torch.load(f"../models/{args.model}", map_location="cpu")
    except Exception:
        logger.error(f"The provided filename {args.model} is invalid")
        return

    model = torch.quantization.convert(model)
    logger.info("Scripting the model")
    scripted_model = torch.jit.script(model)
    logger.info("Optimizing for mobile")
    opt_model = mobile_optimizer.optimize_for_mobile(scripted_model)
    output_filename = (
        args.output if args.output else f"../models/mobile_{args.model[:-4]}.ptl"
    )
    logger.info(f"Outputing the file to {output_filename}")
    opt_model._save_for_lite_interpreter(output_filename)


if __name__ == "__main__":
    main()
