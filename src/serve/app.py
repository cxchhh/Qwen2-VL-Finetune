import argparse
import os
import sys
from threading import Thread
import gradio as gr
from PIL import Image

from transformers import TextIteratorStreamer
from functools import partial
import warnings
from qwen_vl_utils import process_vision_info

from src.utils import load_pretrained_model, get_model_name_from_path, disable_torch_init

warnings.filterwarnings("ignore")

os.environ['GRADIO_TEMP_DIR'] = os.path.expanduser("~/.tmp")
os.environ['GRADIO_ALLOWED_PATHS'] = os.path.expanduser("~")

def is_video_file(filename):
    video_extensions = ['.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm', '.mpeg']
    return any(filename.lower().endswith(ext) for ext in video_extensions)

def bot_streaming(message, history, generation_args):
    # Initialize variables
    images = []
    videos = []

    if message["files"]:
        for file_item in message["files"]:
            if isinstance(file_item, dict):
                file_path = file_item["path"]
            else:
                file_path = file_item
            if is_video_file(file_path):
                videos.append(file_path)
            else:
                images.append(file_path)

    conversation = []
    for user_turn, assistant_turn in history:
        user_content = []
        if isinstance(user_turn, tuple):
            file_paths = user_turn[0]
            user_text = user_turn[1]
            if not isinstance(file_paths, list):
                file_paths = [file_paths]
            for file_path in file_paths:
                if is_video_file(file_path):
                    user_content.append({"type": "video", "video": file_path, "fps":1.0})
                else:
                    user_content.append({"type": "image", "image": file_path})
            if user_text:
                user_content.append({"type": "text", "text": user_text})
        else:
            user_content.append({"type": "text", "text": user_turn})
        conversation.append({"role": "user", "content": user_content})

        if assistant_turn is not None:
            assistant_content = [{"type": "text", "text": assistant_turn}]
            conversation.append({"role": "assistant", "content": assistant_content})

    user_content = []
    for image in images:
        user_content.append({"type": "image", "image": image})
    for video in videos:
        user_content.append({"type": "video", "video": video, "fps":1.0})
    user_text = message['text']
    if user_text:
        user_content.append({"type": "text", "text": user_text})
    conversation.append({"role": "user", "content": user_content})

    prompt = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(conversation)
    
    inputs = processor(text=[prompt], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(device) 

    streamer = TextIteratorStreamer(processor.tokenizer, **{"skip_special_tokens": True, "skip_prompt": True, 'clean_up_tokenization_spaces':False,}) 
    generation_kwargs = dict(inputs, streamer=streamer, eos_token_id=processor.tokenizer.eos_token_id, **generation_args)

    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    buffer = ""
    for new_text in streamer:
        buffer += new_text
        # yield buffer
    thread.join()
    return buffer

def bot_generate_once(message, history, generation_args):
    images = []
    videos = []

    if message["files"]:
        for file_item in message["files"]:
            if isinstance(file_item, dict):
                file_path = file_item["path"]
            else:
                file_path = file_item
            if is_video_file(file_path):
                videos.append(file_path)
            else:
                images.append(file_path)

    conversation = []
    for user_turn, assistant_turn in history:
        user_content = []
        if isinstance(user_turn, tuple):
            file_paths = user_turn[0]
            user_text = user_turn[1]
            if not isinstance(file_paths, list):
                file_paths = [file_paths]
            for file_path in file_paths:
                if is_video_file(file_path):
                    user_content.append({"type": "video", "video": file_path, "fps": 1.0})
                else:
                    user_content.append({"type": "image", "image": file_path})
            if user_text:
                user_content.append({"type": "text", "text": user_text})
        else:
            user_content.append({"type": "text", "text": user_turn})
        conversation.append({"role": "user", "content": user_content})

        if assistant_turn is not None:
            assistant_content = [{"type": "text", "text": assistant_turn}]
            conversation.append({"role": "assistant", "content": assistant_content})

    user_content = []
    for image in images:
        user_content.append({"type": "image", "image": image})
    for video in videos:
        user_content.append({"type": "video", "video": video, "fps": 1.0})
    user_text = message["text"]
    if user_text:
        user_content.append({"type": "text", "text": user_text})
    conversation.append({"role": "user", "content": user_content})

    prompt = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(conversation)

    inputs = processor(
        text=[prompt], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt"
    ).to(device)
    
    generation_kwargs = dict(inputs, eos_token_id=processor.tokenizer.eos_token_id, **generation_args)
    output_ids = model.generate(**generation_kwargs)
    output_text = processor.tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    return output_text

def main(args):

    global processor, model, device

    device = args.device
    
    disable_torch_init()

    use_flash_attn = True
    
    model_name = get_model_name_from_path(args.model_path)
    
    if args.disable_flash_attention:
        use_flash_attn = False

    processor, model = load_pretrained_model(model_base = args.model_base, model_path = args.model_path, 
                                                device_map=args.device, model_name=model_name, 
                                                load_4bit=args.load_4bit, load_8bit=args.load_8bit,
                                                device=args.device, use_flash_attn=use_flash_attn
    )

    
    
    generation_args = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "do_sample": True if args.temperature > 0 else False,
        "repetition_penalty": args.repetition_penalty,
    }
    
    chatbot = gr.Chatbot(scale=2)
    chat_input = gr.MultimodalTextbox(interactive=True, file_types=["image", "video"], placeholder="Enter message or upload file...",
                                  show_label=False)
    bot_streaming_with_args = partial(bot_streaming, generation_args=generation_args)

    with gr.Blocks(fill_height=True) as demo:
        gr.ChatInterface(
            fn=bot_streaming_with_args,
            title="Qwen2.5-VL-7B Instruct",
            stop_btn="Stop Generation",
            multimodal=True,
            textbox=chat_input,
            chatbot=chatbot,
        )
        # gr.Interface(
        #     fn=partial(bot_generate_once, generation_args=generation_args),
        #     inputs=[
        #         gr.JSON(label="Message"),
        #         # gr.JSON(label="History"),
        #     ],
        #     outputs=gr.Textbox(label="Output"),
        #     api_name="chat"
        # )


    # Based on the scene, reason about the operations needed to complete '{lang}', output in json format

    demo = demo.queue(api_open=True)
    demo.launch(show_api=True, share=False, server_name='localhost', allowed_paths=[os.path.expanduser("~")])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--model-base", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--disable_flash_attention", action="store_true")
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(args)
    