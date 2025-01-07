import os
from datasets import load_dataset
import torch
import json
from transformers import AutoTokenizer, LlamaTokenizer, LlamaForCausalLM, AutoModelForCausalLM
from tqdm import tqdm
import numpy as np
import random
import argparse
from llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn
import torch.distributed as dist
import torch.multiprocessing as mp
from streaming_llm.kv_cache import StartRecentKVCache
from streaming_llm.utils import parse_args, load
from streaming_llm.enable_streaming_llm import enable_streaming_llm, enable_streaming_llm_cam

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='NousResearch/Llama-2-7b-chat-hf')
    parser.add_argument('--e', action='store_true', help="Evaluate on LongBench-E")
    return parser.parse_args(args)

# This is the customized building prompt for chat models
def build_chat(tokenizer, prompt, model_name):
    if "chatglm3" in model_name:
        prompt = tokenizer.build_chat_input(prompt)
    elif "chatglm" in model_name:
        prompt = tokenizer.build_prompt(prompt)
    elif "longchat" in model_name or "vicuna" in model_name:
        from fastchat.model import get_conversation_template
        conv = get_conversation_template("vicuna")
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
    elif "llama2" in model_name:
        prompt = f"[INST]{prompt}[/INST]"
    elif "xgen" in model_name:
        header = (
            "A chat between a curious human and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the human's questions.\n\n"
        )
        prompt = header + f" ### Human: {prompt}\n###"
    elif "internlm" in model_name:
        prompt = f"<|User|>:{prompt}<eoh>\n<|Bot|>:"
    return prompt

def post_process(response, model_name):
    if "xgen" in model_name:
        response = response.strip().replace("Assistant:", "")
    elif "internlm" in model_name:
        response = response.split("<eoa>")[0]
    return response

def get_pred(rank, world_size, data, max_length, max_gen, prompt_format, dataset, device, model_name, model2path, out_path):
    device = torch.device(f'cuda:{rank}')
    # replace_llama_attn_with_flash_attn()
    model, tokenizer = load(model_name)
    # kv_cache = enable_streaming_llm(
    #     model, start_size=4, recent_size=252
    # )
    # model, tokenizer = load_model_and_tokenizer(model2path[model_name], model_name, device)
    # model = AutoModelForCausalLM.from_pretrained(
    #     model_name,
    #     torch_dtype="auto",
    #     device_map="auto"
    # )
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    for json_obj in tqdm(data):
        prompt = prompt_format.format(**json_obj)
        # truncate to fit max_length (we suggest truncate in the middle, since the left and right side may contain crucial instructions)
        tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
        if "chatglm3" in model_name:
            tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt", add_special_tokens=False).input_ids[0]
        if len(tokenized_prompt) > max_length:
            half = int(max_length/2)
            prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True)+tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
        if dataset not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]: # chat models are better off without build prompts on these tasks
            prompt = build_chat(tokenizer, prompt, model_name)
        if "chatglm3" in model_name:
            if dataset in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]:
                input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
            else:
                input = prompt.to(device)
        else:
            input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
        context_length = input.input_ids.shape[-1]

        past_key_values = None
        # if kv_cache is not None:
        #     space_needed = 0.3*context_length + max_gen
        #     # space_needed = context_length + 10000
        #     past_key_values = kv_cache.evict_for_space(past_key_values, space_needed)

        if dataset == "samsum": # prevent illegal output on samsum (model endlessly repeat "\nDialogue"), might be a prompting issue
            pred = greedy_generate(model, tokenizer, input.input_ids, max_gen, past_key_values=past_key_values, eos_token_id=[tokenizer.eos_token_id, tokenizer.encode("\n", add_special_tokens=False)[-1]])
            # output = model.generate(
            #     **input,
            #     max_new_tokens=max_gen,
            #     num_beams=1,
            #     do_sample=False,
            #     temperature=1.0,
            #     min_length=context_length+1,
            #     eos_token_id=[tokenizer.eos_token_id, tokenizer.encode("\n", add_special_tokens=False)[-1]],
            # )[0]
        else:
            pred = greedy_generate(model, tokenizer, input.input_ids, max_gen, past_key_values=past_key_values, eos_token_id=[tokenizer.eos_token_id, tokenizer.encode("\n", add_special_tokens=False)[-1]])
            # output = model.generate(
            #     **input,
            #     max_new_tokens=max_gen,
            #     num_beams=1,
            #     do_sample=False,
            #     temperature=1.0,
            # )[0]
        # pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)
        pred = post_process(pred, model_name)
        with open(out_path, "a", encoding="utf-8") as f:
            json.dump({"pred": pred, "answers": json_obj["answers"], "all_classes": json_obj["all_classes"], "length": json_obj["length"]}, f, ensure_ascii=False)
            f.write('\n')
    dist.destroy_process_group()

def greedy_generate(model, tokenizer, input_ids, max_gen_len, past_key_values=None, eos_token_id=None):
    from streaming_llm.pos_shift.modify_llama_cam import enable_llama_pos_shift_attention_cam
    enable_llama_pos_shift_attention_cam(model)
    outputs = model(
        input_ids=input_ids,
        past_key_values=past_key_values,
        use_cache=True,
    )
    past_key_values = outputs.past_key_values
    pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
    generated_ids = [pred_token_idx.item()]
    pos = 0
    for i in range(max_gen_len - 1):
        outputs = model(
            input_ids=pred_token_idx,
            past_key_values=past_key_values,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values
        if i == 0:
            context_length = int(1*past_key_values[0][0].shape[2]) # 50% budget
            # context_length = 0.35*past_key_values[0][0].shape[1] # 35% budget
            kv_cache = enable_streaming_llm_cam(
                model, start_size=4, recent_size=context_length - 4
            )

            if kv_cache is not None:
                print("kvsize:", past_key_values[0][0].shape)
                past_key_values = kv_cache(past_key_values) # __call__()
                print("kvsize:", past_key_values[0][0].shape)

        pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
        generated_ids.append(pred_token_idx.item())
        generated_text = (
            tokenizer.decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
                spaces_between_special_tokens=False,
            )
            .strip()
            .split(" ")
        )

        # now = len(generated_text) - 1
        # if now > pos:
        #     print(" ".join(generated_text[pos:now]), end=" ", flush=True)
        #     pos = now

        if pred_token_idx in eos_token_id:
            break


    text = " ".join(generated_text[:])
    print(text)
    return text

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

# def load_model_and_tokenizer(path, model_name, device):
#     if "chatglm" in model_name or "internlm" in model_name or "xgen" in model_name:
#         tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
#         model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device)
#     elif "llama2" in model_name:
#         replace_llama_attn_with_flash_attn()
#         model, tokenizer = load(path)
#     elif "longchat" in model_name or "vicuna" in model_name:
#         from fastchat.model import load_model
#         replace_llama_attn_with_flash_attn()
#         model, _ = load_model(
#             path,
#             device='cpu',
#             num_gpus=0,
#             load_8bit=False,
#             cpu_offloading=False,
#             debug=False,
#         )
#         model = model.to(device)
#         model = model.bfloat16()
#         tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
#     model = model.eval()
#     return model, tokenizer

if __name__ == '__main__':
    seed_everything(42)
    args = parse_args()
    world_size = torch.cuda.device_count()
    mp.set_start_method('spawn', force=True)

    model2path = json.load(open("config/model2path.json", "r"))
    model2maxlen = json.load(open("config/model2maxlen.json", "r"))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_name = args.model
    # define your model
    max_length = 3500
    if args.e:
        datasets = ["qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "gov_report", "multi_news", \
            "trec", "triviaqa", "samsum", "passage_count", "passage_retrieval_en", "lcc", "repobench-p"]
    else:
        datasets = ["narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh", "hotpotqa", "2wikimqa", "musique", \
                    "dureader", "gov_report", "qmsum", "multi_news", "vcsum", "trec", "triviaqa", "samsum", "lsht", \
                    "passage_count", "passage_retrieval_en", "passage_retrieval_zh", "lcc", "repobench-p"]

    # datasets = ["qasper"]
    # we design specific prompt format and max generation length for each task, feel free to modify them to optimize model output
    dataset2prompt = json.load(open("config/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open("config/dataset2maxlen.json", "r"))
    # predict on each dataset
    if not os.path.exists("pred"):
        os.makedirs("pred")
    if not os.path.exists("pred_e"):
        os.makedirs("pred_e")
    compression = 1
    for dataset in datasets:
        print(dataset)
        if args.e:
            data = load_dataset('THUDM/LongBench', f"{dataset}_e", split='test')
            if not os.path.exists(f"pred_e/{model_name}"):
                os.makedirs(f"pred_e/{model_name}")
            out_path = f"pred_e/{model_name}/{dataset}.jsonl"
        else:
            data = load_dataset('THUDM/LongBench', dataset, split='test')
            if not os.path.exists(f"pred/{model_name}_{compression}"):
                os.makedirs(f"pred/{model_name}_{compression}")
            out_path = f"pred/{model_name}_{compression}/{dataset}.jsonl"
        prompt_format = dataset2prompt[dataset]
        max_gen = dataset2maxlen[dataset]
        data_all = [data_sample for data_sample in data]
        # data_subsets = [data_all[i::world_size] for i in range(world_size)]
        # processes = []
        # for rank in range(world_size):
        #     p = mp.Process(target=get_pred, args=(rank, world_size, data_subsets[rank], max_length, \
        #                 max_gen, prompt_format, dataset, device, model_name, model2path, out_path))
        #     p.start()
        #     processes.append(p)
        # for p in processes:
        #     p.join()
        get_pred(0, 1, data_all, max_length, max_gen, prompt_format, dataset, device, model_name, model2path, out_path)