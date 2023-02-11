import os
import pydoc
from pathlib import Path
from typing import Type

from utils.model import ModelHyperParameters, ModelBase


# Based on gluonts.shell.serve
def model_type_by_name(name: str) -> Type[ModelBase]:
    model_module = pydoc.locate(f"ml.models.{name}")

    if model_module is None:
        raise ValueError(f"Cannot locate model with classname [{name}]")

    print(f"Using model [{model_module}]")
    return getattr(model_module, "Model")


if __name__ == '__main__':
    """
    To quickly iterate, you can run this via cli with `python3 -m model --dataset_dir ../out/datasets --model_dir ../out/local_cli/model`.
    This assumes that you have a valid dataset, which can be created via the train notebook
    """
    args = ModelHyperParameters().parse_args()
    if not args.dataset_dir:
        args.dataset_dir = os.environ['SM_CHANNEL_DATASET']
    if not args.model_dir:
        args.model_dir = os.environ['SM_MODEL_DIR']

    model_output_dir_path = Path(args.model_dir)
    model_output_dir_path.mkdir(parents=True, exist_ok=True)

    model = model_type_by_name(args.model_type)
    model(args).train()
