import click
import pathlib
from typing import Optional


@click.group()
def cli():
    """Florence Tool CLI for processing images with Florence-2 model."""
    pass


@cli.command()
@click.option(
    "--hf-hub-or-path",
    type=str,
    required=True,
    help="Path or Hugging Face hub model identifier for Florence-2.",
)
@click.option(
    "--device",
    type=str,
    default="cuda:0",
    help='Device to run the model on (e.g., "cuda:0" or "cpu").',
)
@click.option(
    "--dtype",
    type=str,
    default="float16",
    help='Torch dtype to use, e.g., "float16" or "float32".',
)
@click.option(
    "--task",
    type=str,
    required=True,
    help='Task to run (e.g., "<CAPTION>", "<OD>", etc.).',
)
@click.option(
    "--image",
    type=click.Path(exists=True, dir_okay=False, path_type=pathlib.Path),
    help="Path to an image file.",
)
@click.option(
    "--folder",
    type=click.Path(exists=True, file_okay=False, path_type=pathlib.Path),
    help="Path to a folder containing images.",
)
@click.option(
    "--output-dir",
    type=click.Path(exists=True, file_okay=False, path_type=pathlib.Path),
    help="Directory to save the results.",
)
@click.option(
    "--text-input",
    type=str,
    default=None,
    help="Optional text input for tasks that require it.",
)
@click.option(
    "--max-new-tokens",
    type=int,
    default=1024,
    help="Maximum number of new tokens to generate.",
)
@click.option(
    "--num-beams", type=int, default=3, help="Number of beams for beam search."
)
@click.option(
    "--output-format",
    type=click.Choice(["json", "csv", "txt"]),
    default="json",
    help="Format to save the results.",
)
@click.option("--recursive", is_flag=True, help="Process sub-directories.")
@click.option(
    "--suffix", type=str, default=None, help="Suffix to use for the output file."
)
@click.option(
    "--overwrite",
    is_flag=True,
    help="Overwrite existing files or append/update them.",
)
@click.option(
    "--image-extensions",
    type=str,
    default="jpg,png",
    help="Comma-separated list of image file extensions to include, e.g., 'jpg,png,jpeg'.",
)
@click.option("--batch-size", type=int, default=1, help="Batch size.")
def run(
    hf_hub_or_path: str,
    device: str,
    dtype: str,
    task: str,
    image: Optional[pathlib.Path],
    folder: Optional[pathlib.Path],
    output_dir: Optional[pathlib.Path],
    text_input: str,
    max_new_tokens: int,
    num_beams: int,
    output_format: str,
    recursive: bool,
    suffix: str,
    overwrite: bool,
    image_extensions: str,
    batch_size: int,
):
    from florence_tool import FlorenceTool

    extensions = image_extensions.split(",")

    tool = FlorenceTool(hf_hub_or_path=hf_hub_or_path, device=device, dtype=dtype)

    tool.load_model()

    if image:
        tool.file(
            path=image,
            task=task,
            text_input=text_input,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            output_dir=output_dir,
            output_format=output_format,
            suffix=suffix,
            overwrite=overwrite,
        )
        click.echo(f"Processed image {image}. Results saved to {output_dir}.")

    elif folder:
        tool.folder(
            folder_path=folder,
            task=task,
            text_input=text_input,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            output_dir=output_dir,
            output_format=output_format,
            recursive=recursive,
            suffix=suffix,
            overwrite=overwrite,
            image_extensions=extensions,
            batch_size=batch_size,
        )
        click.echo(
            f"Processed images in folder {folder}. Results saved to {output_dir}."
        )

    else:
        click.echo("Please specify either an image file or a folder of images.")

    tool.unload_model()


if __name__ == "__main__":
    cli()
