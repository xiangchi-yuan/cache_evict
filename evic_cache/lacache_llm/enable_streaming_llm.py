from streaming_llm.kv_cache import StartRecentKVCache
from streaming_llm.kv_cache_cam import StartRecentKVCache_cam

def enable_streaming_llm(model, start_size, recent_size):
    if "llama" in model.config.model_type:
        k_seq_dim = v_seq_dim = 2
        from streaming_llm.pos_shift.modify_llama import (
            enable_llama_pos_shift_attention,
        )

        enable_llama_pos_shift_attention(model)
    elif "mpt" in model.config.model_type:
        v_seq_dim = 2
        k_seq_dim = 3
    elif "gpt_neox" in model.config.model_type:
        k_seq_dim = v_seq_dim = 2
        from streaming_llm.pos_shift.modify_gpt_neox import (
            enable_gpt_neox_pos_shift_attention,
        )

        enable_gpt_neox_pos_shift_attention(model)
    elif "falcon" in model.config.model_type:
        v_seq_dim = 1
        k_seq_dim = 1
        from streaming_llm.pos_shift.modify_falcon import (
            enable_falcon_pos_shift_attention,
        )

        enable_falcon_pos_shift_attention(model)
    else:
        raise ValueError(f"got {model.config.model_type}")

    # if 'cam' in kv_type:
    #     kv_cache = StartRecentKVCache_cam(
    #         start_size=start_size,
    #         recent_size=recent_size,
    #         k_seq_dim=k_seq_dim,
    #         v_seq_dim=v_seq_dim,
    #     )
    # else:
    kv_cache = StartRecentKVCache(
        start_size=start_size,
        recent_size=recent_size,
        k_seq_dim=k_seq_dim,
        v_seq_dim=v_seq_dim,
    )
    return kv_cache


def enable_streaming_llm_cam(model, start_size, recent_size):
    k_seq_dim = v_seq_dim = 2
    from streaming_llm.pos_shift.modify_llama_cam import (
        enable_llama_pos_shift_attention_cam,
    )

    enable_llama_pos_shift_attention_cam(model)


    # if 'cam' in kv_type:
    #     kv_cache = StartRecentKVCache_cam(
    #         start_size=start_size,
    #         recent_size=recent_size,
    #         k_seq_dim=k_seq_dim,
    #         v_seq_dim=v_seq_dim,
    #     )
    # else:
    kv_cache = StartRecentKVCache_cam(
        start_size=start_size,
        recent_size=recent_size,
        k_seq_dim=k_seq_dim,
        v_seq_dim=v_seq_dim,
    )
    return kv_cache