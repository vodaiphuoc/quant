
def run_quant():
    import requests
    import torch
    import os
    from PIL import Image
    from loguru import logger
    from transformers import AutoProcessor, Gemma3ForConditionalGeneration, AutoModelForCausalLM
    from datasets import load_dataset
    from llmcompressor import oneshot
    from llmcompressor.modifiers.smoothquant import SmoothQuantModifier
    from llmcompressor.modifiers.quantization import GPTQModifier, QuantizationModifier
    from compressed_tensors.quantization import (
        QuantizationArgs
    )
    from llmcompressor.transformers.compression.helpers import calculate_offload_device_map, custom_offload_device_map


    # Load model.
    model_id = "google/gemma-3-4b-it"

    model = Gemma3ForConditionalGeneration.from_pretrained(
        model_id, 
        device_map = "auto",
        torch_dtype = "auto"
    )
    processor = AutoProcessor.from_pretrained(
        model_id, 
        trust_remote_code=True
    )

    # Oneshot arguments
    NUM_CALIBRATION_SAMPLES = 32
    MAX_SEQUENCE_LENGTH = 512
    DATASET_ID = "lmms-lab/flickr30k"
    DATASET_SPLIT = "test"


    # Define a oneshot data collator for multimodal inputs.
    def data_collator(batch):
        assert len(batch) == 1
        return {
            key: torch.tensor(value) \
            if key != "pixel_values" \
            else torch.tensor(value, dtype = torch.int32)
            for key, value in batch[0].items()
        }

    # Load dataset and preprocess.
    ds = load_dataset(DATASET_ID, split=f"{DATASET_SPLIT}[:{NUM_CALIBRATION_SAMPLES}]")
    ds = ds.shuffle(seed=42)


    # Apply chat template and tokenize inputs.
    def preprocess_and_tokenize(example):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": example["image"]},
                    {"type": "text", "text": "What does this image show?"},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": " ".join(example["caption"])},
                ],
            },
        ]
        
        # tokenize
        return processor.apply_chat_template(
            messages, 
            add_generation_prompt=False, 
            tokenize=True,
            return_dict=True, 
            return_tensors="pt",
            max_length=MAX_SEQUENCE_LENGTH,
            padding=False,
            truncation=True,
        )

    ds = ds.map(preprocess_and_tokenize, remove_columns=ds.column_names)

    try:

        # Perform oneshot
        oneshot(
            model=model,
            tokenizer=model_id,
            dataset=ds,
            # splits={"calibration": "test[:16]"},
            recipe= os.path.join(os.path.dirname(__file__), "recipe.yaml"),
            max_seq_length=MAX_SEQUENCE_LENGTH,
            num_calibration_samples=NUM_CALIBRATION_SAMPLES,
            trust_remote_code_model=True,
            data_collator=data_collator,
            output_dir=model_id+"_FP8DYNAMIC"
        )
    except Exception as e:
        logger.error("error in oneshot: {}".format(e))

if __name__ == "__main__":
    run_quant()