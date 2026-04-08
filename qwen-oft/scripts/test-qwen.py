from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
import torch
# torch.set_printoptions(threshold=100000000)
# default: Load the model on the available device(s)
model = Qwen3VLForConditionalGeneration.from_pretrained(
    "/mnt/cpfs/chenhao/huggingface/hub/models--Qwen--Qwen3-VL-4B-Instruct/snapshots/ebb281ec70b05090aa6165b016eac8ec08e71b17", dtype=torch.bfloat16, trust_remote_code=True, local_files_only=True
)
model.to("cuda:0")


# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
# model = Qwen3VLForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen3-VL-4B-Instruct",
#     dtype=torch.bfloat16,
#     attn_implementation="flash_attention_2",
#     device_map="auto",
# )

processor = AutoProcessor.from_pretrained("/mnt/cpfs/chenhao/huggingface/hub/models--Qwen--Qwen3-VL-4B-Instruct/snapshots/ebb281ec70b05090aa6165b016eac8ec08e71b17", local_files_only=True)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "/mnt/cpfs/chenhao/DoubleRL-VLA/training_data/rlbench/close_box/episode0/image0.png",
            },
            {"type": "text", "text": "Describe this image."},
        ],
    }
]

# Preparation for inference
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt"
)
inputs = inputs.to(model.device)

print(list(inputs.keys()))

# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=1280)
print(inputs.input_ids)
print(generated_ids)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)
