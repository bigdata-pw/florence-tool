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
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import sys

IS_WINDOWS = sys.platform == "win32"

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


class FlorenceDataset(Dataset):
    def __init__(
        self,
        images: List[Union[str, pathlib.Path]],
        task_prompt: TASK_TYPE,
        device: Union[str, torch.device],
        dtype: Union[str, torch.dtype],
        processor: Florence2Processor,
        text_inputs: Optional[Union[str, List[str]]] = None,
        convert_rgb: bool = True,
        transform=None,
    ):
        self.images = images
        self.tasks = [task_prompt] * len(images)
        if isinstance(text_inputs, str):
            self.text_inputs = [text_inputs] * len(images)
        elif isinstance(text_inputs, list):
            self.text_inputs = text_inputs
        else:
            self.text_inputs = [None] * len(images)
        self.processor = processor
        self.device = device
        self.dtype = dtype
        self.transform = transform
        self.convert_rgb = convert_rgb

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx])
        if self.convert_rgb:
            image = image.convert("RGB")
        if self.transform:
            image = self.transform(image)

        task_prompt = self.tasks[idx]
        text_input = self.text_inputs[idx]
        prompt = task_prompt
        if text_input is not None:
            prompt = task_prompt + text_input

        model_inputs = self.processor(text=prompt, images=image, return_tensors="pt")
        model_inputs = {
            "input_ids": model_inputs["input_ids"],
            "pixel_values": model_inputs["pixel_values"],
            "width": image.width,
            "height": image.height,
            "path": self.images[idx],
        }
        return model_inputs

    @staticmethod
    def collate_fn(batch):
        batch_inputs = {
            "input_ids": [],
            "pixel_values": [],
            "width": [],
            "height": [],
            "path": [],
        }
        for item in batch:
            for key in batch_inputs:
                batch_inputs[key].append(item[key])
        batch_inputs = {
            key: (
                torch.cat(value, dim=0) if isinstance(value[0], torch.Tensor) else value
            )
            for key, value in batch_inputs.items()
        }
        return batch_inputs


class FlorenceTool:
    def __init__(
        self,
        hf_hub_or_path: str,
        device: Union[str, torch.device],
        dtype: Union[str, torch.dtype],
        max_workers: int = 8,
    ) -> None:
        self.set_device(device)
        self.set_dtype(dtype)
        self.hf_hub_or_path = hf_hub_or_path
        self.model = None
        self.processor = None
        self.save_executor = ThreadPoolExecutor(max_workers=max_workers)
        self.save_futures: List[Future] = []
        self.post_process_executor = ThreadPoolExecutor(max_workers=max_workers)
        self.post_process_futures: List[Future] = []
        self.dataloader = None
        self.dataset = None
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

    def unload_model(self):
        self.wait_for_post_process()
        self.wait_for_save()
        self.model = None
        self.processor = None

    def __del__(self):
        self.wait_for_post_process()
        self.wait_for_save()

    def wait_for_post_process(self):
        if len(self.post_process_futures) == 0:
            return
        logging.info("Waiting for post processing to complete.")

        for future in self.post_process_futures:
            future.result()

        self.post_process_executor.shutdown(wait=True)
        self.post_process_futures.clear()
        logging.info("Post processing complete.")

    def wait_for_save(self):
        if len(self.save_futures) == 0:
            return
        logging.info("Waiting for save to complete.")

        for future in self.save_futures:
            future.result()

        self.save_executor.shutdown(wait=True)
        self.save_futures.clear()
        logging.info("Saving complete.")

    def get_data_loader(
        self,
        image_paths: List[Union[str, pathlib.Path]],
        task_prompt: TASK_TYPE,
        text_inputs: Optional[Union[str, List[str]]] = None,
        batch_size: int = 4,
        num_workers: int = 4,
        convert_rgb: bool = True,
    ):
        if IS_WINDOWS:
            num_workers = 0
            logging.info("Windows environment: Overriding `num_workers` to 0.")
        logging.info("Creating FlorenceDataset.")
        self.dataset = FlorenceDataset(
            images=image_paths,
            task_prompt=task_prompt,
            device=self.device,
            dtype=self.dtype,
            processor=self.processor,
            text_inputs=text_inputs,
            convert_rgb=convert_rgb,
        )
        logging.info(f"FlorenceDataset created - found {len(self.dataset)} images.")
        logging.info("Creating DataLoader.")
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
            collate_fn=FlorenceDataset.collate_fn,
        )
        logging.info("DataLoader created.")

    def run_dataloader(
        self,
        images: List[Union[str, pathlib.Path]],
        task_prompt: TASK_TYPE,
        text_inputs: Optional[Union[str, List[str]]] = None,
        max_new_tokens: Optional[int] = 1024,
        num_beams: Optional[int] = 3,
        batch_size: int = 4,
        num_workers: int = 4,
        save_result: bool = True,
        output_format: OUTPUT_TYPE = "json",
        output_dir: Optional[Union[str, pathlib.Path]] = None,
        suffix: Optional[str] = None,
        overwrite: bool = True,
        convert_rgb: bool = True,
    ):
        if self.model is None or self.processor is None:
            logging.error("Call `load_model` before `run`.")
            return
        assert (
            task_prompt in TASK_TYPES
        ), f"{task_prompt} is not supported. Expected one of {TASK_TYPES}."

        if isinstance(images, Image.Image):
            images = [images]

        self.get_data_loader(
            image_paths=images,
            task_prompt=task_prompt,
            text_inputs=text_inputs,
            batch_size=batch_size,
            num_workers=num_workers,
            convert_rgb=convert_rgb,
        )

        with tqdm(total=len(self.dataset)) as pbar:
            for batch in self.dataloader:
                generated_ids = self.model.generate(
                    input_ids=batch["input_ids"].to(self.device),
                    pixel_values=batch["pixel_values"].to(self.device, self.dtype),
                    max_new_tokens=max_new_tokens,
                    num_beams=num_beams,
                )
                future = self.post_process_executor.submit(
                    self.post_process_batch,
                    generated_ids=generated_ids,
                    task_prompt=task_prompt,
                    width=batch["width"],
                    height=batch["height"],
                    image_paths=batch["path"],
                    save_result=save_result,
                    output_format=output_format,
                    output_dir=output_dir,
                    suffix=suffix,
                    overwrite=overwrite,
                )
                self.post_process_futures.append(future)
                pbar.update(len(batch["path"]))

    def run(
        self,
        image: Image.Image,
        task_prompt: TASK_TYPE,
        text_input: Optional[str] = None,
        max_new_tokens: Optional[int] = 1024,
        num_beams: Optional[int] = 3,
    ):
        """
        Run the model on one image.

        :param images: A single PIL image.
        :param task_prompt: The task prompt.
        :param text_inputs: A single text input.
        :param max_new_tokens: Maximum number of new tokens to generate.
        :param num_beams: Number of beams for beam search.
        :return: A single result.
        """
        if self.model is None or self.processor is None:
            logging.error("Call `load_model` before `run`.")
            return
        assert (
            task_prompt in TASK_TYPES
        ), f"{task_prompt} is not supported. Expected one of {TASK_TYPES}."

        prompt = task_prompt
        if text_input is not None:
            prompt = task_prompt + text_input

        inputs = self.processor(text=prompt, images=image, return_tensors="pt")

        generated_ids = self.model.generate(
            input_ids=inputs["input_ids"].to(self.device),
            pixel_values=inputs["pixel_values"].to(self.device, self.dtype),
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
        )

        generated_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=False
        )[0]

        parsed_answer = self.processor.post_process_generation(
            generated_text,
            task=task_prompt,
            image_size=(image.width, image.height),
        )
        return parsed_answer

    def file(
        self,
        path: Union[str, pathlib.Path],
        task: TASK_TYPE,
        text_input: Optional[str] = None,
        max_new_tokens: Optional[int] = 1024,
        num_beams: Optional[int] = 3,
        convert_rgb: bool = True,
        save_result: bool = True,
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
        )
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
        num_workers: int = 4,
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

        self.run_dataloader(
            images=image_paths,
            task_prompt=task,
            text_inputs=text_input,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            batch_size=batch_size,
            save_result=save_result,
            output_format=output_format,
            output_dir=output_dir,
            suffix=suffix,
            overwrite=overwrite,
            convert_rgb=convert_rgb,
            num_workers=num_workers,
        )

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

        output_file = (
            output_dir
            / f"{image_path.stem}-{image_path.suffix}_{suffix}.{output_format}"
        )

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

    def post_process_batch(
        self,
        generated_ids: torch.Tensor,
        task_prompt: TASK_TYPE,
        width: List[int],
        height: List[int],
        image_paths: List[Union[str, pathlib.Path]],
        save_result: bool = True,
        output_format: OUTPUT_TYPE = "json",
        output_dir: Optional[Union[str, pathlib.Path]] = None,
        suffix: Optional[str] = None,
        overwrite: bool = True,
    ):
        generated_texts = self.processor.batch_decode(
            generated_ids, skip_special_tokens=False
        )
        parsed_answers = []
        for i, generated_text in enumerate(generated_texts):
            parsed_answer = self.processor.post_process_generation(
                generated_text,
                task=task_prompt,
                image_size=(width[i], height[i]),
            )
            if save_result:
                future = self.save_executor.submit(
                    self.save_to_file,
                    result=parsed_answer,
                    image_path=image_paths[i],
                    task=task_prompt,
                    output_dir=output_dir,
                    output_format=output_format,
                    suffix=suffix,
                    overwrite=overwrite,
                )
                self.save_futures.append(future)
            parsed_answers.append(parsed_answer)
        return parsed_answers
