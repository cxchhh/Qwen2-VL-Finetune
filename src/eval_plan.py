import argparse
from functools import partial
from itertools import chain
import json
from pathlib import Path
from threading import Thread

import tqdm
import numpy as np
from omegaconf import DictConfig
from transformers import TextIteratorStreamer, AutoProcessor
from torch.utils.tensorboard import SummaryWriter

from qwen_vl_utils import process_vision_info
import yaml
from training.data import SupervisedJudgeDataset
from utils import disable_torch_init, get_model_name_from_path, load_pretrained_model

def bot_streaming(images, prompt, processor:AutoProcessor,  generation_args):
    conversation = []
    user_content = [{"type": "image", "image": image} for image in images]
    user_content.append({
        "type": "text",
        "text": prompt
    })
    conversation.append({"role": "human", "content": user_content})
    chat = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    
    image_inputs, video_inputs = process_vision_info(conversation)

    inputs = processor(
        text=[chat], images=image_inputs, padding=True, return_tensors="pt"
    ).to(device)

    streamer = TextIteratorStreamer(
        processor.tokenizer,
        skip_special_tokens=True,
        skip_prompt=True,
        clean_up_tokenization_spaces=False,
    )
    generation_kwargs = dict(inputs, streamer=streamer, eos_token_id=processor.tokenizer.eos_token_id, **generation_args)

    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    buffer = ""
    for new_text in streamer:
        buffer += new_text

    thread.join()
    return buffer

obs_config = DictConfig({
    "rgb_obs": ["rgb_static", "rgb_gripper"],
    "depth_obs": [],
    "state_obs": ["robot_obs"],
    "actions": ["rel_actions"],
    "language": ["language"],
})

def load_tasks(yaml_path):
    with open(yaml_path, 'r') as f:
        task_dict = yaml.safe_load(f)
    return task_dict

def classify_task(task_sentence, task_yaml):
    for task_name, variations in task_yaml.items():
        if task_sentence.lower().strip() in [v.lower().strip() for v in variations]:
            return task_name
    return task_sentence.replace("_"," ")

def main(args):
    global model, device
    device = args.device

    disable_torch_init()

    use_flash_attn = not args.disable_flash_attention
    model_name = get_model_name_from_path(args.model_path)

    processor, model = load_pretrained_model(
        model_base=args.model_base,
        model_path=args.model_path,
        device_map=args.device,
        model_name=model_name,
        load_4bit=args.load_4bit,
        load_8bit=args.load_8bit,
        device=args.device,
        use_flash_attn=use_flash_attn,
    )

    processor: AutoProcessor = processor
    print("model loaded")

    generation_args = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "do_sample": args.temperature > 0,
        "repetition_penalty": args.repetition_penalty,
    }

    bot_streaming_with_args = partial(bot_streaming, generation_args=generation_args, processor=processor)

    # ✅ TensorBoard writer
    writer = SummaryWriter(log_dir="./eval_log")

    task_yaml = load_tasks("/mnt/afs/chenxuchuan/datasets/calvin/new_playtable.yaml")


    with open(Path(args.eval_dataset), "r") as f:
        judge_data = json.load(f)
        print("judge data loaded")

    calvin_datasets_dir = Path(args.calvin_datasets_dir)
    try:
        lang_data = np.load(calvin_datasets_dir / "lang_annotations" / "auto_lang_ann.npy", allow_pickle=True).item()
    except Exception:
        lang_data = np.load(calvin_datasets_dir / "auto_lang_ann.npy", allow_pickle=True).item()

    keys = list(chain(*obs_config.values()))
    keys.remove('language')
    keys.append("scene_obs")

    anns = lang_data['language']['ann']
    indx = lang_data["info"]["indx"]
    print("calvin data loaded")

    tot = acc = err = 0
    for data in tqdm.tqdm(judge_data[1000:]):
        global_id = data["image_id"][0]
        id = int(global_id) // 1000
        frame_id = int(global_id) % 1000
        res = data["conversations"][1]['value']
        prompt = data["conversations"][0]['value']
        episodes = [
            np.load((calvin_datasets_dir / f"episode_{file_idx:07d}.npz").as_posix())
            for file_idx in range(*indx[id])
        ]
        episode = {key: np.stack([ep[key] for ep in episodes]) for key in keys}
        n_frames = episode['rgb_static'].shape[0]

        task = classify_task(anns[id], task_yaml)

        subtasks = prompt.split("?")[1]

        rgb_primary = episode['rgb_static'][min(frame_id, n_frames - 1)]
        rgb_gripper = episode['rgb_gripper'][min(frame_id, n_frames - 1)]

        rgb_primary_bytes = SupervisedJudgeDataset.numpy_to_base64(rgb_primary)
        rgb_gripper_bytes = SupervisedJudgeDataset.numpy_to_base64(rgb_gripper)

        images = [rgb_primary_bytes, rgb_gripper_bytes]

        judgement = bot_streaming_with_args(images, prompt)

        

        if judgement.strip().replace("the ", "") != res.strip().replace("the ", ""):
            err += 1
            icon = '❌'
        else:
            acc += 1
            icon = '✅'

        tot += 1

        print(task, subtasks,",", judgement, res, icon)

        if tot % 100 == 0:
            acc_rate = acc / tot * 100
            err_rate = err / tot * 100
            print(
                f"[{tot}] ✅ {acc:6d} ({acc_rate:5.2f}%) | ❌ {err:6d} ({err_rate:5.2f}%)\n",
            )
            # ✅ TensorBoard logging
            writer.add_scalar("Accuracy/Correct", acc_rate, tot)
            writer.add_scalar("Error/Err", err_rate, tot)

    writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--model-base", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--calvin_datasets_dir", type=str, default="/mnt/afs/chenxuchuan/datasets/calvin/task_ABC_D/validation/")
    parser.add_argument("--eval_dataset", type=str, default="/mnt/afs/chenxuchuan/datasets/qwen_ft/plan_val.json")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--disable_flash_attention", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    if args.debug:
        args.calvin_datasets_dir = "/mnt/afs/chenxuchuan/datasets/calvin/calvin_debug_dataset/validation/"
        args.eval_dataset = "/mnt/afs/chenxuchuan/datasets/qwen_ft/debug/plan_val.json"
    main(args)
