import copy
import argparse
import torch
from tqdm import tqdm
import torch.nn as nn
import copy
def main():
    torch.manual_seed(1)
    parser = argparse.ArgumentParser(description='token matching')
    parser.add_argument('--checkpoint', type=str, help='parent checkpoint')
    parser.add_argument('--output', type=str, help='path to write the embeddings to')
    parser.add_argument('--parent-dict', type=str, help='parent side dictionary')
    parser.add_argument('--child-dict', type=str, help='child side dict')
    parser.add_argument('--switch-dict', type=str, choices=['src', 'tgt'] )

    args = parser.parse_args()
    model = torch.load(args.checkpoint)

    if args.switch_dict == 'tgt':
        parent_embeddings = model['model']['decoder.embed_tokens.weight']
    elif args.switch_dict == 'src':
        parent_embeddings = model['model']['encoder.embed_tokens.weight']
    parent_dict_file = open(args.parent_dict, 'r')
    parent_dict_lines = parent_dict_file.readlines()
    child_dict_file = open(args.child_dict, 'r')
    child_dict_lines = child_dict_file.readlines()

    p_dict_list = ['<s>','<pad>','</s>','<unk>']
    c_dict_list = ['<s>','<pad>','</s>','<unk>']

    for line in tqdm(parent_dict_lines):
        type = line.strip('\n').split()[0]
        p_dict_list.append(type)
    
    for line in tqdm(child_dict_lines):
        type = line.strip('\n').split()[0]
        c_dict_list.append(type)
    
    inter_list = list(set(p_dict_list).intersection(set(c_dict_list)))
    p_tensor_dict = {}
    for idx in tqdm(range(len(p_dict_list))):
        p_tensor_dict[p_dict_list[idx]] = parent_embeddings[idx]
    embedding_dim=int(parent_embeddings.size()[-1])
    child_embeddings = torch.empty(len(c_dict_list), embedding_dim)
    nn.init.normal_(child_embeddings, mean=0, std=embedding_dim**-0.5)
    cnt = 0
    old_child_embeddings = copy.deepcopy(child_embeddings)
    for i in tqdm(range(len(c_dict_list))):
        if c_dict_list[i] in p_dict_list:
            child_embeddings[i] = p_tensor_dict[c_dict_list[i]]
            cnt += 1
    print(cnt)
    # assert old_child_embeddings != child_embeddings

    new_checkpoint = copy.deepcopy(model)
    if 'cfg' not in new_checkpoint.keys():
        model_cfg_dict = vars(new_checkpoint['args'])
        model_cfg_dict['share_decoder_input_output_embed'] = True
        model_cfg_dict['share_all_embeddings'] = False
    else:
        model_cfg_dict = vars(new_checkpoint['cfg']['model'])
        model_cfg_dict['share_decoder_input_output_embed'] = True
        model_cfg_dict['share_all_embeddings'] = False
    if 'cfg' not in new_checkpoint.keys():
        new_checkpoint['args'] = argparse.Namespace(**model_cfg_dict)
    else:
        new_checkpoint['cfg']['model'] = argparse.Namespace(**model_cfg_dict)
    if args.switch_dict == 'tgt':
        new_checkpoint['model']['decoder.embed_tokens.weight'] = child_embeddings
    elif args.switch_dict == 'src':
        new_checkpoint['model']['encoder.embed_tokens.weight'] = child_embeddings
    torch.save(new_checkpoint, args.output)

if __name__ == '__main__':
    main()
    