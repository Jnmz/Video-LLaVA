import math
import os
import argparse
import json

import torch
import transformers
from tqdm import tqdm
from videollava.conversation import conv_templates, SeparatorStyle
from videollava.constants import DEFAULT_IM_START_TOKEN, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_END_TOKEN, IMAGE_TOKEN_INDEX, DEFAULT_VID_START_TOKEN, DEFAULT_VID_END_TOKEN
from videollava.mm_utils import get_model_name_from_path, tokenizer_image_token, KeywordsStoppingCriteria
from videollava.model.builder import load_pretrained_model
from videollava.model.language_model.llava_llama import LlavaLlamaForCausalLM
from videollava.train.train import smart_tokenizer_and_embedding_resize
import time
import random 

import numpy as np



def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
set_seed(43)

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

# def parse_args():
#     """
#     Parse command-line arguments.
#     """
#     parser = argparse.ArgumentParser()

#     # Define the command-line arguments
#     parser.add_argument('--model_path', help='', required=True)
#     parser.add_argument('--cache_dir', help='', required=True)
#     parser.add_argument('--video_dir', help='Directory containing video files.', required=True)
#     parser.add_argument('--gt_file_question', help='Path to the ground truth file containing question.', required=True)
#     parser.add_argument('--gt_file_answers', help='Path to the ground truth file containing answers.', required=True)
#     parser.add_argument('--output_dir', help='Directory to save the model results JSON.', required=True)
#     parser.add_argument('--output_name', help='Name of the file for storing results JSON.', required=True)
#     parser.add_argument("--num_chunks", type=int, default=1)
#     parser.add_argument("--chunk_idx", type=int, default=0)
#     parser.add_argument("--device", type=str, required=False, default='cuda:0')
#     parser.add_argument('--model_base', help='', default=None, type=str, required=False)
#     parser.add_argument("--model_max_length", type=int, required=False, default=2048)

#     return parser.parse_args()

def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser()

    # Define the command-line arguments
    parser.add_argument('--model_path', default="/root/autodl-tmp/checkpoints/Video-LLaVA-7B") # adjust accordingly 
    parser.add_argument('--cache_dir', default="/root/autodl-tmp/cache/cache_dir") # adjust accordingly 
    parser.add_argument('--video_dir', default="videollava/eval/GPT_Zero_Shot_QA/MSVD_Zero_Shot_QA/videos",help='Directory containing video files.')
    parser.add_argument('--gt_file_question', default="videollava/eval/GPT_Zero_Shot_QA/MSVD_Zero_Shot_QA/test_q.json" ,help='Path to the ground truth file containing question.')
    parser.add_argument('--gt_file_answers', default="videollava/eval/GPT_Zero_Shot_QA/MSVD_Zero_Shot_QA/test_a.json",help='Path to the ground truth file containing answers.')
    parser.add_argument('--output_dir', default="videollava/eval/GPT_Zero_Shot_QA/Video-LLaVA-7B",help='Directory to save the model results JSON.')
    parser.add_argument('--output_name', default="1_0", help='Name of the file for storing results JSON.')
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--num_images", type=int, default=8)
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--device", type=str, required=False, default='cuda')
    parser.add_argument('--model_base', help='', default=None, type=str, required=False)
    parser.add_argument("--model_max_length", type=int, required=False, default=2048)

    return parser.parse_args()

def get_model_output(model, video_processor, tokenizer, video, qs, args):
    if model.config.mm_use_im_start_end:
        qs = DEFAULT_VID_START_TOKEN + ''.join([DEFAULT_IMAGE_TOKEN]*args.num_images) + DEFAULT_VID_END_TOKEN + '\n' + qs
    else:
        qs = ''.join([DEFAULT_IMAGE_TOKEN]*args.num_images) + '\n' + qs

    conv_mode = "llava_v1"
    args.conv_mode = conv_mode

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()


    video_tensor = video_processor.preprocess(video, return_tensors='pt')['pixel_values'][0].half().to(args.device)
    # print(video_tensor.shape)
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(args.device)

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
    # with torch.inference_mode():
    #     tower = model.get_video_tower()
    #     print(tower)
        
    # with torch.inference_mode():
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            images=[video_tensor],
            do_sample=True,
            temperature=1.0, # originally 0.0
            max_new_tokens=1024,
            use_cache=True,
            stopping_criteria=[stopping_criteria])

    input_token_len = input_ids.shape[1]
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[:-len(stop_str)]
    outputs = outputs.strip()
    # print(outputs)
    return outputs


def run_inference(args):
    """
    Run inference on ActivityNet QA DataSet using the Video-ChatGPT model.

    Args:
        args: Command-line arguments.
    """
    # Initialize the model
    model_name = get_model_name_from_path(args.model_path)
    # print(torch.cuda.device_count())
    tokenizer, model, processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name)
    

    model = model.to(args.device)


    gt_questions = json.load(open(args.gt_file_question, "r"))
    gt_questions = get_chunk(gt_questions, args.num_chunks, args.chunk_idx)
    gt_answers = json.load(open(args.gt_file_answers, "r"))

    answers_file = os.path.join(args.output_dir, f"{args.output_name}.json")
    os.makedirs(args.output_dir, exist_ok=True)
    ans_file = open(answers_file, "w")

    # Create the output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    output_list = []  # List to store the output results


    video_formats = ['.mp4', '.avi', '.mov', '.mkv']

    # Iterate over each sample in the ground truth file
    index = 10
    sample = gt_questions[index]
    video_name = sample['video_name']
    question = sample['question']
    id = sample['question_id']
    answer = gt_answers[index]['answer']
    fmt = '.avi'
    video_path = os.path.join(args.video_dir, f"{video_name}{fmt}")
    
    t0 = time.time()
    output = get_model_output(model, processor['video'], tokenizer, video_path, question, args)
    te = time.time()

    print("=========================")
    print(output)
    print("total = " + str(te-t0))
    print("=========================")
    # 1.91 2.07 3.21 1.85 1.85
    # 1.88 2.10 3.20 
    # Save the output list to a JSON file
    # with open(os.path.join(args.output_dir, f"{args.output_name}.json"), 'w') as file:
    #     json.dump(output_list, file)


if __name__ == "__main__":
    args = parse_args()
    run_inference(args)
