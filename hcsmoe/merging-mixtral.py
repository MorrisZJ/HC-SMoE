# -*- coding: utf-8 -*-
# @Author: pingzhili
# @Time: 2024/2/18

# modified
# @Author: wazenmai
# @Time: 2024/8/13

import os
import sys
import json
import time
import shutil
import torch
import pickle
import logging
import itertools
from fire import Fire
from tqdm import tqdm
from typing import Optional, List, Union
from transformers import MixtralForCausalLM, AutoTokenizer

import hcsmoe.models.mixtral  # noqa: F401  — registers hcsmoe_mixtral with HF Auto classes
from hcsmoe.models.mixtral import MixtralConfig as HCSMoEMixtralConfig

# Path to the HC-SMoE Mixtral model source files; copied alongside weights so that
# AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True) works without
# any prior import or manual registration.
_HCSMOE_MIXTRAL_SRC_DIR = os.path.dirname(
    os.path.abspath(hcsmoe.models.mixtral.__file__)
)

from hcsmoe.evaluation import evaluate_fewshot, get_calib_dataloder
from hcsmoe.merging.grouping_mixtral import ExpertsGrouperForMixtral
from hcsmoe.merging.grouping_mixtral import merge_by_groups_with_usage_weighted, merge_by_groups_within_and_across_models

logger = logging.getLogger(__name__)

from typing import Optional


###############################
#  Helper class and functions #
###############################
class Args:
    def __init__(
        self,
        task,
        num_average_groups: int,
        model_name: Optional[str] = "mistralai/Mixtral-8x7B-v0.1",
        dominant: Optional[str] = "knowledge",
        similarity_base: Optional[str] = "router-logits",
        merge: Optional[str] = "zipit",
        mode: Optional[str] = "normal",
        n_sentences: Optional[int] = 32,
        train_batch_size: Optional[int] = 4,
        eval_batch_size: Optional[int] = 32,
        partition: Optional[int] = 1,
        start_layer: Optional[int] = 0,
        output_path: Optional[str] = None,
        result_path: Optional[str] = None,
        model_path: Optional[str] = None,
        group_limit: Optional[int] = 4,
        data_limit: Optional[int] = 50000,
        num_fewshot: Optional[int] = 0,
        random_start_center: Optional[bool] = False,
        weight= None,
        cluster: Optional[str] = "kmeans",
        linkage: Optional[str] = "ward",
        hierarchical_stopping_metric: Optional[str] = "silhouette",
        overlap_metric: Optional[str] = "cosine",
        dynamic_group: Optional[bool] = False,
    ):
        self.task = task
        self.num_average_groups = num_average_groups
        self.model_name = model_name
        self.dominant = dominant
        self.similarity_base = similarity_base
        self.merge = merge
        self.mode = mode
        self.n_sentences = n_sentences
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.partition = partition
        self.start_layer = start_layer
        self.output_path = output_path
        self.result_path = result_path
        self.model_path = model_path
        self.group_limit = group_limit
        self.data_limit = data_limit
        self.num_fewshot = num_fewshot
        self.random_start_center = random_start_center
        self.weight = weight
        self.cluster = cluster
        self.linkage = linkage
        self.hierarchical_stopping_metric = hierarchical_stopping_metric
        self.overlap_metric = overlap_metric
        self.dynamic_group = dynamic_group

def get_dataloader(args, tokenizer):
    return get_calib_dataloder(
        dataset="c4",
        tokenizer=tokenizer,
        max_block_size=2048,
        n_blocks_for_stat=args.n_sentences, # 32, 128
        batch_size=args.train_batch_size,
        num_workers=4,
    )

def get_grouper(args, config):
    return ExpertsGrouperForMixtral(
        config=config,
        similarity_base=args.similarity_base,
        start_layer=args.start_layer,
        group_limit=args.group_limit,
        data_limit=args.data_limit,
        random_start_center=args.random_start_center,
        cluster=args.cluster,
        linkage=args.linkage,
        hierarchical_stopping_metric=args.hierarchical_stopping_metric,
        overlap_metric=args.overlap_metric,
        dynamic_group=args.dynamic_group,
    )

def evaluation(args, model, tokenizer):
    if args.task == "no":
        return

    result_dir = args.result_path.split("/")[:-1]
    result_dir = "/".join(result_dir)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    if isinstance(args.task, str):
        evaluate_fewshot(
            model, tokenizer=tokenizer, task=args.task, num_fewshot=args.num_fewshot, output_path=args.result_path, log=True
        )
    else:
        for i, t in enumerate(args.tasks):
            evaluate_fewshot(
                model, tokenizer=tokenizer, task=t, num_fewshot=args.num_fewshot, eval_batch_size=args.eval_batch_size, output_path=args.result_path, log=True
            )

def print_usage_frequency(usage_dict):
    for k in usage_dict:
        for num in usage_dict[k]:
            print(round(num.item(), 4), end=',')
        print()


def _build_hcsmoe_config(model) -> HCSMoEMixtralConfig:
    """Build a HCSMoEMixtralConfig that exactly matches the current model architecture.

    This is called after merging/pruning so that save_pretrained writes a config whose
    num_local_experts and num_experts_per_tok fields reflect the *actual* (potentially
    modified) architecture rather than the original HF hub values.
    """
    num_layers = model.config.num_hidden_layers

    expert_counts: List[int] = []
    gate_counts: List[int] = []
    for i in range(num_layers):
        layer = model.model.layers[i]
        moe = getattr(layer, "block_sparse_moe", None)
        if moe is None:
            # Layer has no MoE block (layer_experts_idx=None case)
            expert_counts.append(0)
            gate_counts.append(0)
        else:
            n_exp = len(moe.experts)
            expert_counts.append(n_exp)
            gate_counts.append(moe.gate.out_features)

    # Use an int when all layers are uniform; otherwise a list (handles dynamic pruning)
    all_same_experts = len(set(expert_counts)) == 1
    num_local_experts: Union[int, List[int]] = expert_counts[0] if all_same_experts else expert_counts

    all_same_gate = len(set(gate_counts)) == 1
    gate_num_experts: Union[int, List[int], None]
    if set(gate_counts) == set(expert_counts):
        # gate output matches expert count on every layer — no need to store separately
        gate_num_experts = None
    else:
        gate_num_experts = gate_counts[0] if all_same_gate else gate_counts

    # Clamp num_experts_per_tok so it cannot exceed the smallest non-zero expert count
    non_zero = [c for c in expert_counts if c > 0]
    min_experts = min(non_zero) if non_zero else 1
    num_experts_per_tok = min(model.config.num_experts_per_tok, min_experts)

    # Build the config dict from the existing config and override changed fields.
    base_dict = model.config.to_dict()
    # Remove keys that we will supply explicitly, plus internal PretrainedConfig bookkeeping.
    for _key in ("transformers_version", "model_type",
                 "num_local_experts", "num_experts_per_tok", "gate_num_experts"):
        base_dict.pop(_key, None)

    new_config = HCSMoEMixtralConfig(
        **base_dict,
        num_local_experts=num_local_experts,
        num_experts_per_tok=num_experts_per_tok,
        gate_num_experts=gate_num_experts,
    )
    # auto_map lets HuggingFace load the custom model code from the saved directory
    # without any prior import, i.e.:
    #   AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True)
    new_config.auto_map = {
        "AutoConfig": "configuration_mixtral.MixtralConfig",
        "AutoModelForCausalLM": "modeling_mixtral.MixtralForCausalLM",
    }
    return new_config


###############################
###      Main function      ###
###############################
def run_hcsmoe(
        task: str,
        num_average_groups: int,
        model_name: Optional[str] = "mistralai/Mixtral-8x7B-v0.1",
        dominant: Optional[str] = "knowledge", # random, frequency, no
        similarity_base: Optional[str] = "router-logits", # router-logits, weight, expert-output
        merge: Optional[str] = "zipit", # no, freq, zipit, kl-weight, fix-dom-same
        mode: Optional[str] = "normal", # normal, activation-with-router-logits, input-weight, all
        n_sentences: Optional[int] = 32,
        train_batch_size: Optional[int] = 4,
        eval_batch_size: Optional[int] = 32,
        partition: Optional[int] = 1,
        start_layer: Optional[int] = 0,
        output_path: Optional[str] = None,
        result_path: Optional[str] = None,
        model_path: Optional[str] = None,
        group_limit: Optional[int] = 4,
        data_limit: Optional[int] = 50000,
        num_fewshot: Optional[int] = 0,
        random_start_center: Optional[bool] = False,
        cluster: Optional[str] = "kmeans",
        linkage: Optional[str] = "ward",
        hierarchical_stopping_metric: Optional[str] = "silhouette",
        ingredient: Optional[str] = "act", # act, weight, act+weight
        overlap_metric: Optional[str] = "cosine", # kl-divergence, wasserstein, cosine,
        dynamic_group: Optional[bool] = False,
):
    print(f"Merge model {model_name} with {num_average_groups} group, {dominant} dominant + {similarity_base} grouping + {merge} {mode} merge with ingredient {ingredient}, evaluate on {task}")
    print(f"Cluster: {cluster}, linkage: {linkage}, hierarchical_stopping_metric: {hierarchical_stopping_metric}, overlap_metric: {overlap_metric}, dynamic_group: {dynamic_group}")


    ### 1. Initialization
    args = Args(
        task=task,
        num_average_groups=num_average_groups,
        model_name=model_name,
        dominant=dominant,
        similarity_base=similarity_base,
        merge=merge,
        mode=mode,
        n_sentences=n_sentences,
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
        partition=partition,
        start_layer=start_layer,
        output_path=output_path,
        result_path=result_path,
        model_path=model_path,
        group_limit=group_limit,
        data_limit=data_limit,
        num_fewshot=num_fewshot,
        random_start_center=random_start_center,
        cluster=cluster,
        linkage=linkage,
        hierarchical_stopping_metric=hierarchical_stopping_metric,
        overlap_metric=overlap_metric,
        dynamic_group=dynamic_group,
    )
    
    torch.manual_seed(0)
    eval_ppl = (task == "minipile")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    model = MixtralForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16, device_map="auto"
    )
    if model_path:
        model.load_state_dict(torch.load(model_path))
    model.eval()
    dataloader_for_merging = get_dataloader(args, tokenizer)

    grouper = get_grouper(args, model.config)

    
    print("[HC-SMoE] Number of parameters before merging:", model.num_parameters())
    print(f"[HC-SMoE] Merging into average {num_average_groups} groups...")
    group_st = time.time()
    if merge == "freq" or dominant == "frequency" or mode == "freq":
        grouper.compute_all_usages(model, dataloader_for_merging)
        print_usage_frequency(grouper._usage_frequency_state_dict)
    if dynamic_group:
        grouper.compute_all_usages(model, dataloader_for_merging, mode=hierarchical_stopping_metric)
        print_usage_frequency(grouper._usage_frequency_state_dict)
    

    ### 2. Get dominant experts
    dom_experts = None
    if merge == "fsm" or merge == "no":
        pass
    elif dominant == "random":
        grouper.group_experts_randomly(num_groups=args.num_average_groups)
        dom_experts = None
    elif dominant == "frequency":
        grouper.compute_all_similarities(model, dataloader_for_merging)
        dom_experts = grouper.group_experts_globally_from_dominant_experts(
            num_average_groups=num_average_groups, merging_layers=list(range(start_layer, model.config.num_hidden_layers))
        )
    elif dominant == "routing-score":
        grouper.compute_all_usages(model, dataloader_for_merging, mode="routing-score")
        print_usage_frequency(grouper._usage_frequency_state_dict)
        dom_experts = grouper.group_experts_globally_from_dominant_experts(
            num_average_groups=num_average_groups, merging_layers=list(range(start_layer, model.config.num_hidden_layers))
        )
    elif dominant == "no":
        dom_experts = grouper.cluster_experts(model=model, dataloader=dataloader_for_merging, num_groups=num_average_groups)
    else:
        raise ValueError(
            f"Accepted dominant are `random`, `frequency`, `no`, but the input is `{dominant}`")

    
    ### 3. Merge experts
    if merge == "no":
        pass
    elif merge == "freq":
        model = merge_by_groups_with_usage_weighted(
            model, grouper=grouper, merging_layers=list(range(start_layer, model.config.num_hidden_layers))
        )
    else:
        model = merge_by_groups_within_and_across_models(
            mixtral_model=model,
            grouper=grouper,
            dataloader=dataloader_for_merging,
            merge=merge,
            mode=mode,
            partition=partition,
            dominant_alone=False,
            core_experts=dom_experts,
            ingredient=ingredient,
        )
    
    print(f"[HC-SMoE] Merging time: {time.time() - group_st:2f} seconds")
    
    
    if merge != "no":
        ### 4. Print grouping results
        print(f"[HC-SMoE] ========= Grouping results ========= ")
        for name, state in grouper.group_state_dict().items():
            if dom_experts is None:
                print(f"Group {name}: {state.tolist()}")
            else:
                print(f"Group {name}: {state.tolist()} (DOMs are {dom_experts[name]})")
        del grouper

    if merge == "unmerge":
        print(f"[HC-SMoE] ======= Grouping of unmerge ======= ")
        for layer_idx in range(start_layer, model.config.num_hidden_layers):
            print(f"--- Layer {layer_idx} ---")
            print(f"expert_to_group: {model.model.layers[layer_idx].block_sparse_moe.expert_to_group}")
            print(f"group_to_expert: {model.model.layers[layer_idx].block_sparse_moe.group_to_expert}")
            print(f"unmerge_matrix: {model.model.layers[layer_idx].block_sparse_moe.unmerge_matrix}")
    
    ### 5. Save the model
    print("[HC-SMoE] Number of parameters after merging:", model.num_parameters())

    # Build an HCSMoEMixtralConfig that reflects the actual (post-merge/prune) architecture.
    # This ensures from_pretrained initialises the model with the right number of experts per
    # layer and never randomly-initialises any weight.
    hcsmoe_config = _build_hcsmoe_config(model)
    model.config = hcsmoe_config

    os.makedirs(output_path, exist_ok=True)
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    # Copy the HC-SMoE Mixtral source files so that the saved directory is self-contained.
    # Any environment that has `transformers` installed can then load the model with:
    #   AutoModelForCausalLM.from_pretrained(output_path, trust_remote_code=True)
    for _fname in ("modeling_mixtral.py", "configuration_mixtral.py"):
        shutil.copy(
            os.path.join(_HCSMOE_MIXTRAL_SRC_DIR, _fname),
            os.path.join(output_path, _fname),
        )
    print(f"[HC-SMoE] Model and tokenizer saved to {output_path} (HuggingFace format).")
    torch.cuda.empty_cache()

    ### 6. Evaluation
    evaluation(args, model, tokenizer)

if __name__ == "__main__":
    Fire(run_hcsmoe)
