# Florence Tool CLI

## Overview

The **Florence Tool CLI** provides a command-line interface for processing images using the [Florence-2](https://huggingface.co/microsoft/Florence-2-large) model. This tool allows users to apply various visual and text-based tasks, such as object detection, captioning, and OCR, on individual images or entire folders of images.

## Features

- **Model Loading:** Load and run the Florence-2 model from a local path or Hugging Face hub.
- **Task Variety:** Supports a wide range of tasks, including captioning, object detection, dense region captioning, OCR, and more.
- **Batch Processing:** Efficiently process images in batches from a folder.
- **Recursive Search:** Optionally process images within subdirectories.
- **Customizable Output:** Save results in JSON, CSV, or plain text formats with optional suffixes and overwrite modes.
- **Flexible Image Handling:** Specify the image file extensions to process, allowing for flexibility in file types.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/bigdata-pw/florence-tool.git
   cd florence-tool
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Install the tool:

   ```bash
   pip install -e .
   ```

## Usage

### Command-Line Interface

You can use the tool directly from the command line by running the following command:

```bash
florence-tool run [OPTIONS]
```

### Options

- `--hf-hub-or-path` (required): Path or Hugging Face hub model identifier for Florence-2.
- `--device`: Device to run the model on (e.g., `"cuda:0"` or `"cpu"`). Default is `"cuda:0"`.
- `--dtype`: Torch dtype to use (e.g., `"float16"`, `"float32"`, `"bfloat16"`). Default is `"float16"`.
- `--task` (required): Task to run (e.g., `"<CAPTION>"`, `"<OD>"`, etc.).
- `--image`: Path to an image file.
- `--folder`: Path to a folder containing images.
- `--wds`: WebDataset.
- `--output-dir`: Directory to save the results.
- `--text-input`: Optional text input for tasks that require it.
- `--max-new-tokens`: Maximum number of new tokens to generate. Default is `1024`.
- `--num-beams`: Number of beams for beam search. Default is `3`.
- `--output-format`: Format to save the results (`json`, `csv`, or `txt`). Default is `json`.
- `--recursive`: Process subdirectories if specified.
- `--suffix`: Suffix to use for the output file.
- `--overwrite`: Flag to overwrite existing files. If not specified, appends/updates the files.
- `--image-extensions`: Comma-separated list of image file extensions to include (e.g., `"jpg,png,jpeg"`). Default is `"jpg,png"`.
- `--batch-size`: Number of images to process in a batch. Default is `1`.
- `--num-workers`: Number of Dataloader workers. Default is `4`, overriden to `0` on Windows.
- `--image-key`: WebDataset image key.

### Examples

#### Processing a Single Image

```bash
florence-tool run --hf-hub-or-path microsoft/Florence-2-large --task "<CAPTION>" --image /path/to/image.jpg --output-dir /path/to/output/
```

#### Processing Images in a Folder

```bash
florence-tool run --hf-hub-or-path microsoft/Florence-2-large --task "<OD>" --folder /path/to/folder/ --output-dir /path/to/output/
```

#### Processing a WebDataset

```bash
florence-tool run --hf-hub-or-path microsoft/Florence-2-large --task "<CAPTION>" --wds "shard-{00000..00069}.tar" --output-dir /path/to/output/
```

#### Processing a WebDataset (streaming)

```bash
florence-tool run --hf-hub-or-path microsoft/Florence-2-large --task "<CAPTION>" --wds "pipe:aws s3 cp s3://data/shard-{00000..00069}.tar -" --output-dir /path/to/output/
```

```bash
florence-tool run --hf-hub-or-path microsoft/Florence-2-large --task "<CAPTION>" --wds "pipe:aws s3 cp s3://data/shard-{00000..00069}.tar --endpoint-url https://00000000000000000000000000000000.r2.cloudflarestorage.com -" --output-dir /path/to/output/
```

#### Processing Images Recursively

```bash
florence-tool run --hf-hub-or-path microsoft/Florence-2-large --task "<OCR>" --folder /path/to/folder/ --output-dir /path/to/output/ --recursive
```

#### Custom Image Extensions

```bash
florence-tool run --hf-hub-or-path microsoft/Florence-2-large --task "<DENSE_REGION_CAPTION>" --folder /path/to/folder/ --image-extensions jpg,png,jpeg --output-dir /path/to/output/
```

#### Saving Results in CSV Format

```bash
florence-tool run --hf-hub-or-path microsoft/Florence-2-large --task "<REGION_PROPOSAL>" --folder /path/to/folder/ --output-dir /path/to/output/ --output-format csv
```

#### Overwriting Existing Files

```bash
florence-tool run --hf-hub-or-path microsoft/Florence-2-large --task "<CAPTION>" --folder /path/to/folder/ --output-dir /path/to/output/ --suffix captions --overwrite
```

#### Processing in Batches

```bash
florence-tool run --hf-hub-or-path microsoft/Florence-2-large --task "<CAPTION>" --folder /path/to/folder/ --output-dir /path/to/output/ --batch-size 4
```

#### Running on CPU

```bash
florence-tool run --hf-hub-or-path microsoft/Florence-2-large --task "<CAPTION>" --folder /path/to/folder/ --output-dir /path/to/output/ --device "cpu"
```

### Tasks

```
<OCR>
<OCR_WITH_REGION>
<CAPTION>
<DETAILED_CAPTION>
<MORE_DETAILED_CAPTION>
<OD>
<DENSE_REGION_CAPTION>
<CAPTION_TO_PHRASE_GROUNDING>
<REFERRING_EXPRESSION_SEGMENTATION>
<REGION_TO_SEGMENTATION>
<OPEN_VOCABULARY_DETECTION>
<REGION_TO_CATEGORY>
<REGION_TO_DESCRIPTION>
<REGION_TO_OCR>
<REGION_PROPOSAL>
```

### Development

#### Code Structure

- `florence_tool.py`: Main class that implements the Florence-2 model handling and processing logic.
- `cli.py`: Command-line interface built with `Click`.
- `modeling`: Directory containing model configuration and processing scripts.

#### Running Locally

To run the CLI locally without installing:

```bash
python -m florence_tool.cli run [OPTIONS]
```

### Contributing

Contributions are welcome! Please submit a pull request or open an issue if you have ideas or find a bug.

### License

This project is licensed under the Apache 2.0 License.
