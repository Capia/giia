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
    args = ModelHyperParameters().parse_args()

    model_output_dir_path = Path(args.model_dir)
    model_output_dir_path.mkdir(parents=True, exist_ok=True)

    model = model_type_by_name(args.model_type)
    model(args).train()
