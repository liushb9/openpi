import torch
from transformers import AutoModelForCausalLM
from janus.models import MultiModalityCausalLM, VLChatProcessor
from janus.utils.io import load_pil_images
import json
from tqdm import tqdm

model_path = "deepseek-ai/Janus-Pro-1B"
vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
tokenizer = vl_chat_processor.tokenizer

vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(model_path)
vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()


question = f"What action should the robot take to put the carrot on the plate? Please describe the next action using a behavioral description. You can state in which direction the robotic arm should move, or what specific behavior the robotic arm should perform. These should be atomic actions, not long-sequence behaviors."

image = "/gpfs/0607-cluster/chenhao/DoubleRL-VLA/training_data/simpler/PutOnPlateInScene25Single-v1/success_proc_0_numid_0_epsid_0/image0.png"
conversation = [
    {
        "role": "<|User|>",
        "content": f"<image_placeholder>\n{question}",
        "images": [image],
    },
    {"role": "<|Assistant|>", "content": ""},
]

pil_images = load_pil_images(conversation)
prepare_inputs = vl_chat_processor(
    conversations=conversation, images=pil_images, force_batchify=True
).to(vl_gpt.device)

print(prepare_inputs.input_ids)

print(prepare_inputs.attention_mask.shape, prepare_inputs.input_ids.shape)

inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

outputs = vl_gpt.language_model.generate(
    inputs_embeds=inputs_embeds,
    attention_mask=prepare_inputs.attention_mask,
    pad_token_id=tokenizer.eos_token_id,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    max_new_tokens=100,
    do_sample=False,
    temperature=0.6,
    use_cache=True,
)

print(outputs)
answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True).strip()
print(answer)





# json_path = "/gpfs/0607-cluster/chenhao/DoubleRL-VLA/training_data/json/simpler_train.json"
# with open(json_path, "r", encoding="utf-8") as f:
#     data = json.load(f)

# for d in tqdm(data, desc="Generating language plans"):
#     question = f"What action should the robot take to {d['input_prompt']}? Tell me **only the next step** that should be taken."

#     image = d["input_image"][0]
#     conversation = [
#         {
#             "role": "<|User|>",
#             "content": f"<image_placeholder>\n{question}",
#             "images": [image],
#         },
#         {"role": "<|Assistant|>", "content": ""},
#     ]

#     pil_images = load_pil_images(conversation)
#     prepare_inputs = vl_chat_processor(
#         conversations=conversation, images=pil_images, force_batchify=True
#     ).to(vl_gpt.device)

#     inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

#     outputs = vl_gpt.language_model.generate(
#         inputs_embeds=inputs_embeds,
#         attention_mask=prepare_inputs.attention_mask,
#         pad_token_id=tokenizer.eos_token_id,
#         bos_token_id=tokenizer.bos_token_id,
#         eos_token_id=tokenizer.eos_token_id,
#         max_new_tokens=100,
#         do_sample=False,
#         temperature=0,
#         use_cache=True,
#     )

#     answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True).strip()
#     print(answer)
#     d["language_plan"] = answer

# save_path = "/gpfs/0607-cluster/chenhao/DoubleRL-VLA/training_data/json/simpler_train_with_plan_temp_0.json"
# with open(save_path, "w", encoding="utf-8") as f:
#     json.dump(data, f, indent=4, ensure_ascii=False)
