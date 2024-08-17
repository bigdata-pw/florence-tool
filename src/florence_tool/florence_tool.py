import warnings

warnings.filterwarnings("ignore")

import pathlib
import csv
import json
from PIL import Image
import torch
from typing import Any, Dict, Literal, Optional, Union, List
from tqdm import tqdm
import logging
from concurrent.futures import ThreadPoolExecutor, Future

from .modeling import Florence2ForConditionalGeneration, Florence2Processor

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

TASK_TYPES = [
    "<OCR>",
    "<OCR_WITH_REGION>",
    "<CAPTION>",
    "<DETAILED_CAPTION>",
    "<MORE_DETAILED_CAPTION>",
    "<OD>",
    "<DENSE_REGION_CAPTION>",
    "<CAPTION_TO_PHRASE_GROUNDING>",
    "<REFERRING_EXPRESSION_SEGMENTATION>",
    "<REGION_TO_SEGMENTATION>",
    "<OPEN_VOCABULARY_DETECTION>",
    "<REGION_TO_CATEGORY>",
    "<REGION_TO_DESCRIPTION>",
    "<REGION_TO_OCR>",
    "<REGION_PROPOSAL>",
]

TASK_TYPE = Literal[
    "<OCR>",
    "<OCR_WITH_REGION>",
    "<CAPTION>",
    "<DETAILED_CAPTION>",
    "<MORE_DETAILED_CAPTION>",
    "<OD>",
    "<DENSE_REGION_CAPTION>",
    "<CAPTION_TO_PHRASE_GROUNDING>",
    "<REFERRING_EXPRESSION_SEGMENTATION>",
    "<REGION_TO_SEGMENTATION>",
    "<OPEN_VOCABULARY_DETECTION>",
    "<REGION_TO_CATEGORY>",
    "<REGION_TO_DESCRIPTION>",
    "<REGION_TO_OCR>",
    "<REGION_PROPOSAL>",
]

OUTPUT_TYPES = ["json", "csv", "txt"]
OUTPUT_TYPE = Literal["json", "csv", "txt"]

DTYPE_MAP = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}


class FlorenceTool:
    def __init__(
        self,
        hf_hub_or_path: str,
        device: Union[str, torch.device],
        dtype: Union[str, torch.dtype],
        max_workers: int = 4,
    ) -> None:
        self.set_device(device)
        self.set_dtype(dtype)
        self.hf_hub_or_path = hf_hub_or_path
        self.model = None
        self.processor = None
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.save_futures: List[Future] = []
        logging.info(
            f"Initialized\n- hf_hub_or_path: {self.hf_hub_or_path}\n- device: {self.device}\n- dtype: {self.dtype}"
        )

    def set_device(self, device: Union[str, torch.device]):
        if isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

    def set_dtype(self, dtype: Union[str, torch.dtype]):
        if isinstance(dtype, str):
            assert dtype in DTYPE_MAP, f"Expected dtype to be one of {DTYPE_MAP}."
            self.dtype = DTYPE_MAP[dtype]
        else:
            self.dtype = dtype
        if self.device.type == "cpu" and self.dtype != torch.float32:
            logging.info(
                "Running on cpu - overriding provided dtype to `torch.float32`."
            )

    def load_model(self):
        logging.info("Loading model")
        self.model: Florence2ForConditionalGeneration = (
            Florence2ForConditionalGeneration.from_pretrained(
                self.hf_hub_or_path, torch_dtype=self.dtype
            ).to(self.device)
        )
        self.processor: Florence2Processor = Florence2Processor.from_pretrained(
            self.hf_hub_or_path
        )
        logging.info("Model loaded")

    def wait_for_save(self):
        if len(self.save_futures) == 0:
            return
        logging.info("Waiting for save to complete.")

        for future in self.save_futures:
            future.result()

        self.executor.shutdown(wait=True)
        self.save_futures.clear()
        logging.info("Saving complete.")

    def unload_model(self):
        self.wait_for_save()
        self.model = None
        self.processor = None

    def __del__(self):
        self.wait_for_save()

    def run(
        self,
        images: Union[Image.Image, List[Image.Image]],
        task_prompt: TASK_TYPE,
        text_inputs: Optional[Union[str, List[str]]] = None,
        max_new_tokens: Optional[int] = 1024,
        num_beams: Optional[int] = 3,
    ):
        """
        Run the model on one or more images.

        :param images: A single PIL image or a list of PIL images.
        :param task_prompt: The task prompt.
        :param text_inputs: A single text input or a list of text inputs. Must match the length of images if a list.
        :param max_new_tokens: Maximum number of new tokens to generate.
        :param num_beams: Number of beams for beam search.
        :return: A single result or a list of results.
        """
        if self.model is None or self.processor is None:
            logging.error("Call `load_model` before `run`.")
            return
        assert (
            task_prompt in TASK_TYPES
        ), f"{task_prompt} is not supported. Expected one of {TASK_TYPES}."

        if isinstance(images, Image.Image):
            images = [images]

        if text_inputs is None:
            prompts = [task_prompt] * len(images)
        elif isinstance(text_inputs, str):
            prompts = [task_prompt + text_inputs] * len(images)
        else:
            assert len(images) == len(
                text_inputs
            ), "Length of text_inputs must match the number of images."
            prompts = [task_prompt + text_input for text_input in text_inputs]

        inputs = self.processor(
            text=prompts, images=images, return_tensors="pt", padding=True
        ).to(self.device, self.dtype)

        generated_ids = self.model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
        )

        generated_texts = self.processor.batch_decode(
            generated_ids, skip_special_tokens=False
        )
        parsed_answers = []

        for i, generated_text in enumerate(generated_texts):
            parsed_answer = self.processor.post_process_generation(
                generated_text,
                task=task_prompt,
                image_size=(images[i].width, images[i].height),
            )
            parsed_answers.append(parsed_answer)

        return parsed_answers

    def file(
        self,
        path: Union[str, pathlib.Path],
        task: TASK_TYPE,
        text_input: Optional[str] = None,
        max_new_tokens: Optional[int] = 1024,
        num_beams: Optional[int] = 3,
        convert_rgb: bool = True,
        save_result: bool = False,
        output_format: OUTPUT_TYPE = "json",
        output_dir: Optional[Union[str, pathlib.Path]] = None,
        suffix: Optional[str] = None,
        overwrite: bool = True,
    ):
        if isinstance(path, str):
            path = pathlib.Path(path)
        assert path.is_file(), "Expected `path` to be a file."
        image = Image.open(path)
        if convert_rgb:
            image = image.convert("RGB")
        result = self.run(
            image=image,
            task_prompt=task,
            text_input=text_input,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
        )[0]
        if save_result:
            self.save_to_file(
                result=result,
                image_path=path,
                task=task,
                output_dir=output_dir,
                output_format=output_format,
                suffix=suffix,
                overwrite=overwrite,
            )
        return result

    def folder(
        self,
        folder_path: Union[str, pathlib.Path],
        task: TASK_TYPE,
        text_input: Optional[str] = None,
        max_new_tokens: Optional[int] = 1024,
        num_beams: Optional[int] = 3,
        batch_size: int = 4,
        convert_rgb: bool = True,
        save_result: bool = True,
        output_format: OUTPUT_TYPE = "json",
        output_dir: Optional[Union[str, pathlib.Path]] = None,
        recursive: bool = False,
        image_extensions: List[str] = ["jpg", "png"],
        suffix: Optional[str] = None,
        overwrite: bool = True,
    ):
        if isinstance(folder_path, str):
            folder_path = pathlib.Path(folder_path)
        assert folder_path.is_dir(), "Expected `folder_path` to be a directory."

        image_paths = []
        for ext in image_extensions:
            if recursive:
                image_paths.extend(folder_path.rglob(f"*.{ext}"))
            else:
                image_paths.extend(folder_path.glob(f"*.{ext}"))
        results = []

        for i in tqdm(range(0, len(image_paths), batch_size)):
            batch_paths = image_paths[i : i + batch_size]
            images = []
            for path in batch_paths:
                try:
                    image = Image.open(path)
                    if convert_rgb:
                        image = image.convert("RGB")
                    images.append(image)
                except Exception as e:
                    logging.error(f"Error opening {path}: {e}")

            if not images:
                continue

            result_batch = self.run(
                images=images,
                task_prompt=task,
                text_inputs=[text_input] * len(images) if text_input else None,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
            )
            results.extend(result_batch)

            if save_result:
                for image, path, result in zip(images, batch_paths, result_batch):
                    future = self.executor.submit(
                        self.save_to_file,
                        result=result,
                        image_path=path,
                        task=task,
                        output_dir=output_dir,
                        output_format=output_format,
                        suffix=suffix,
                        overwrite=overwrite,
                    )
                    self.save_futures.append(future)

        return results

    def save_to_file(
        self,
        result: Dict[str, Any],
        image_path: pathlib.Path,
        task: TASK_TYPE,
        output_dir: Optional[Union[str, pathlib.Path]],
        output_format: OUTPUT_TYPE = "json",
        suffix: Optional[str] = None,
        overwrite: bool = True,
    ):
        assert (
            output_format in OUTPUT_TYPES
        ), f"Expected output_format to be one of {OUTPUT_TYPES}"
        if output_dir is None:
            output_dir = image_path.parent
        if isinstance(output_dir, str):
            output_dir = pathlib.Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if suffix is None:
            suffix = task.strip("<>").lower()

        output_file = output_dir / f"{image_path.stem}_{suffix}.{output_format}"

        if output_format == "json":
            if output_file.exists() and not overwrite:
                with open(output_file, "r") as f:
                    existing_data = json.load(f)
                existing_data[task] = result[task]
                with open(output_file, "w") as f:
                    json.dump(existing_data, f, indent=2)
            else:
                with open(output_file, "w") as f:
                    json.dump({task: result[task]}, f, indent=2)

        elif output_format == "csv":
            if output_file.exists() and not overwrite:
                with open(output_file, "r", newline="") as f:
                    reader = csv.reader(f)
                    existing_data = list(reader)
                new_data = list(result.items())
                combined_data = {row[0]: row[1] for row in existing_data + new_data}
                with open(output_file, "w", newline="") as f:
                    writer = csv.writer(f)
                    for key, value in combined_data.items():
                        writer.writerow([key, value])
            else:
                with open(output_file, "w", newline="") as f:
                    writer = csv.writer(f)
                    for key, value in result.items():
                        writer.writerow([key, value])

        elif output_format == "txt":
            mode = "a" if not overwrite and output_file.exists() else "w"
            with open(output_file, mode) as f:
                f.write(result[task] + "\n")

        logging.debug(f"Saved result to {output_file}")
