import torch
import torch.nn.functional as F
import math
from typing import Type, Dict, Any, Tuple, Callable, Optional
from einops import rearrange, repeat
import copy

from . import merge
from .utils import isinstance_str, init_generator



def compute_merge(x: torch.Tensor, tome_info: Dict[str, Any]) -> Tuple[Callable, ...]:
    original_h, original_w = tome_info["size"]
    original_tokens = original_h * original_w
    downsample = int(math.ceil(math.sqrt(original_tokens // x.shape[1])))

    args = tome_info["args"]

    if downsample <= args["max_downsample"]:
        w = int(math.ceil(original_w / downsample))
        h = int(math.ceil(original_h / downsample))
        r = int(x.shape[1] * args["ratio"])

        # Re-init the generator if it hasn't already been initialized or device has changed.
        if args["generator"] is None:
            args["generator"] = init_generator(x.device)
        elif args["generator"].device != x.device:
            args["generator"] = init_generator(x.device, fallback=args["generator"])
        
        # If the batch size is odd, then it's not possible for prompted and unprompted images to be in the same
        # batch, which causes artifacts with use_rand, so force it to be off.
        use_rand = False if x.shape[0] % 2 == 1 else args["use_rand"]
        m, u = merge.bipartite_soft_matching_random2d(x, w, h, args["sx"], args["sy"], r, 
                                                    no_rand=not use_rand, generator=args["generator"])
    else:
        m, u = (merge.do_nothing, merge.do_nothing)

    m_a, u_a = (m, u) if args["merge_attn"]      else (merge.do_nothing, merge.do_nothing)
    m_c, u_c = (m, u) if args["merge_crossattn"] else (merge.do_nothing, merge.do_nothing)
    m_m, u_m = (m, u) if args["merge_mlp"]       else (merge.do_nothing, merge.do_nothing)

    return m_a, m_c, m_m, u_a, u_c, u_m  # Okay this is probably not very good

def compute_merge_temp(x: torch.Tensor, tome_info: Dict[str, Any], ratio: float) -> Tuple[Callable, ...]:
    original_t = tome_info["num_frames"]
    original_tokens = original_t
    downsample = original_tokens // x.shape[1]
    # print(f"In compute temp: original tokens {original_tokens}, downsample {downsample}")

    args = tome_info["args"]

    if ratio > 0 and downsample <= args["max_downsample"]:
        t = int(math.ceil(original_t / downsample))
        r = int(x.shape[1] * ratio)

        # Re-init the generator if it hasn't already been initialized or device has changed.
        if args["generator"] is None:
            args["generator"] = init_generator(x.device)
        elif args["generator"].device != x.device:
            args["generator"] = init_generator(x.device, fallback=args["generator"])
        
        # If the batch size is odd, then it's not possible for prompted and unprompted images to be in the same
        # batch, which causes artifacts with use_rand, so force it to be off.
        use_rand = False if x.shape[0] % 2 == 1 else args["use_rand"]
        m, u = merge.bipartite_soft_matching_random1d(x, t, args["sx"], r, 
                                                      no_rand=not use_rand, generator=args["generator"])
    else:
        m, u = (merge.do_nothing, merge.do_nothing)

    m_a, u_a = (m, u) if args["merge_attn"]      else (merge.do_nothing, merge.do_nothing)
    m_c, u_c = (m, u) if args["merge_crossattn"] else (merge.do_nothing, merge.do_nothing)
    m_m, u_m = (m, u) if args["merge_mlp"]       else (merge.do_nothing, merge.do_nothing)

    return m_a, m_c, m_m, u_a, u_c, u_m  # Okay this is probably not very good

def get_frame_merge_func(hidden_states, num_frames, height, width, tome_info, enabled=True):
    if enabled:
        batch_frames, seq_length, channels = hidden_states.shape
        batch_size = batch_frames // num_frames
        hidden_states = hidden_states.reshape(batch_frames, height, width, channels).permute(0, 3, 1, 2)
        # # TODO: downsample spatially to save time, should not be hardcoded
        height_down = height
        width_down = (width // height) * height_down 
        hidden_states = F.interpolate(hidden_states, size=[height_down, width_down], mode="area")
        hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch_size, num_frames, height * width * channels)
        # hidden_states = hidden_states.reshape(batch_size, num_frames, seq_length * channels)
        fm_a, fm_c, fm_m, fu_a, fu_c, fu_m = compute_merge_temp(hidden_states, tome_info, tome_info["args"]["fm_ratio"])
        # hidden_states = hidden_states.reshape(batch_frames, seq_length, channels)
        return fm_a, fm_c, fm_m, fu_a, fu_c, fu_m
    else:
        return merge.do_nothing, merge.do_nothing, merge.do_nothing, merge.do_nothing, merge.do_nothing, merge.do_nothing


def get_token_merge_func(hidden_states, tome_info, enabled=True):
    if enabled:
        _, tm_a, tm_c, tm_m, _, tu_a, tu_c, tu_m = compute_merge(hidden_states, tome_info, tome_info["args"]["tm_ratio"])
        return tm_a, tm_c, tm_m, tu_a, tu_c, tu_m
    else:
        return merge.do_nothing, merge.do_nothing, merge.do_nothing, merge.do_nothing, merge.do_nothing, merge.do_nothing



def make_tome_block(block_class: Type[torch.nn.Module]) -> Type[torch.nn.Module]:
    """
    Make a patched class on the fly so we don't have to import any specific modules.
    This patch applies ToMe to the forward function of the block.
    """

    class ToMeBlock(block_class):
        # Save for unpatching later
        _parent = block_class

        def _forward(self, x: torch.Tensor, context: torch.Tensor = None) -> torch.Tensor:
            m_i, m_a, m_c, m_m, u_i, u_a, u_c, u_m = compute_merge(x, self._tome_info)

            # This is where the meat of the computation happens
            x = u_a(self.attn1(m_a(self.norm1(x)), context=context if self.disable_self_attn else None)) + x
            x = u_c(self.attn2(m_c(self.norm2(x)), context=context)) + x
            x = u_m(self.ff(m_m(self.norm3(x)))) + x

            return x
    
    return ToMeBlock






def make_diffusers_tome_block(block_class: Type[torch.nn.Module]) -> Type[torch.nn.Module]:
    """
    Make a patched class for a diffusers model.
    This patch applies ToMe to the forward function of the block.
    """
    class ToMeBlock(block_class):
        # Save for unpatching later
        _parent = block_class

        def forward(
            self,
            hidden_states,
            attention_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            timestep=None,
            cross_attention_kwargs=None,
            class_labels=None,
        ) -> torch.Tensor:
            # (1) ToMe
            m_i, m_a, m_c, m_m, u_i, u_a, u_c, u_m = compute_merge(hidden_states, self._tome_info)

            if self.use_ada_layer_norm:
                norm_hidden_states = self.norm1(hidden_states, timestep)
            elif self.use_ada_layer_norm_zero:
                norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
                    hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype
                )
            else:
                norm_hidden_states = self.norm1(hidden_states)

            # (2) ToMe m_a
            norm_hidden_states = m_a(norm_hidden_states)

            # 1. Self-Attention
            cross_attention_kwargs = cross_attention_kwargs if cross_attention_kwargs is not None else {}
            attn_output = self.attn1(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
                attention_mask=attention_mask,
                **cross_attention_kwargs,
            )
            if self.use_ada_layer_norm_zero:
                attn_output = gate_msa.unsqueeze(1) * attn_output

            # (3) ToMe u_a
            hidden_states = u_a(attn_output) + hidden_states

            if self.attn2 is not None:
                norm_hidden_states = (
                    self.norm2(hidden_states, timestep) if self.use_ada_layer_norm else self.norm2(hidden_states)
                )
                # (4) ToMe m_c
                norm_hidden_states = m_c(norm_hidden_states)

                # 2. Cross-Attention
                attn_output = self.attn2(
                    norm_hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=encoder_attention_mask,
                    **cross_attention_kwargs,
                )
                # (5) ToMe u_c
                hidden_states = u_c(attn_output) + hidden_states

            # 3. Feed-forward
            norm_hidden_states = self.norm3(hidden_states)
            
            if self.use_ada_layer_norm_zero:
                norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

            # (6) ToMe m_m
            norm_hidden_states = m_m(norm_hidden_states)

            ff_output = self.ff(norm_hidden_states)

            if self.use_ada_layer_norm_zero:
                ff_output = gate_mlp.unsqueeze(1) * ff_output

            # (7) ToMe u_m
            hidden_states = u_m(ff_output) + hidden_states

            return hidden_states

    return ToMeBlock


def make_generative_models_tome_block(block_class: Type[torch.nn.Module]) -> Type[torch.nn.Module]:
    """
    Make a patched class on the fly so we don't have to import any specific modules.
    This patch applies ToMe to the forward function of the block.
    """

    class ToMeBlock(block_class):
        # Save for unpatching later
        _parent = block_class

        def _forward(
            self, x, context=None, additional_tokens=None, n_times_crossframe_attn_in_self=0
        ):
            num_frames = self.num_frames
            height = self.height
            width = self.width
            
            batch_frames, seq_length, channels = x.shape
            batch_size = batch_frames // num_frames
            
            orig_x = x
            x = x.reshape(batch_size, num_frames, seq_length * channels)
            # print("-"*50)
            # print(f"fm_ratio: {self._tome_info['args']['fm_ratio']}")
            # print(f"Before frame_temp {x.shape}")
            fm_a, fm_c, fm_m, fu_a, fu_c, fu_m = compute_merge_temp(x, self._tome_info, self._tome_info["args"]["fm_ratio"])
            x = fm_a(x)
            # print(f"After computing frame merge {x.shape}")
            # print("-"*50)
            num_frames_down = x.shape[1]
            x = x.reshape(batch_size * num_frames_down, seq_length, channels)
            _, tm_a, tm_c, tm_m, _, tu_a, tu_c, tu_m = compute_merge(x, self._tome_info, self._tome_info["args"]["tm_ratio"])
            
            x = orig_x
            
            norm_x = self.norm1(x)
            skip_x = x
            
            # Frame merge for self attention or cross attention
            if self.disable_self_attn:  # Cross attention
                fm, fu = fm_c, fu_c
            else:
                fm, fu = fm_a, fu_a

            norm_x = norm_x.reshape(batch_size, num_frames, seq_length * channels)
            # print("tome block shape before frame merge ", norm_x.shape)
            # print(norm_x.shape)
            norm_x = fm(norm_x)
            # print(norm_x.shape)
            # print("tome block shape after frame merge ", norm_x.shape)
            num_frames_down = norm_x.shape[1]
            norm_x = norm_x.reshape(batch_size * num_frames_down, seq_length, channels)
            
            # Token merge for self attention or cross attention
            if self.disable_self_attn: # Cross attention
                tm, tu = tm_c, tu_c
            else:
                tm, tu = tm_a, tu_a

            # print("tome block shape before token merge ", norm_x.shape)
            # print("-"*50)
            # print(f"Before token merge {norm_x.shape}")
            norm_x = tm(norm_x)
            # print(f"After token merge {norm_x.shape}")
            # print("tome block shape after token merge ", norm_x.shape)
            
            x = self.attn1(
                    norm_x,
                    context=context if self.disable_self_attn else None,
                    additional_tokens=additional_tokens,
                    n_times_crossframe_attn_in_self=n_times_crossframe_attn_in_self
                    if not self.disable_self_attn
                    else 0,
                )
            
            # Un merge Token first and then frame
            x = tu(x)
            x = x.reshape(batch_size, num_frames_down, seq_length * channels)
            x = fu(x).reshape(batch_frames, seq_length, channels)
            x = x + skip_x # Add skip connection to output of attn (self or cross) attn
            
            skip_x = x # Store the x -> skip_x 
            norm_x = self.norm2(x)
            
            # Frame merge for self attention or cross attention
            if context is not None:
                fm, fu = fm_c, fu_c
            else:
                fm, fu = fm_a, fu_a

            norm_x = norm_x.reshape(batch_size, num_frames, seq_length * channels)
            norm_x = fm(norm_x)
            num_frames_down = norm_x.shape[1]
            norm_x = norm_x.reshape(batch_size * num_frames_down, seq_length, channels)
            
            # Token merge for self attention or cross attention
            if context is not None:
                tm, tu = tm_c, tu_c
            else:
                tm, tu = tm_a, tu_a
            
            norm_x = tm(norm_x)
            shape = context.shape if context is not None else None
            if context is not None:
                context = context.reshape(batch_size, num_frames, shape[1], shape[2])[:, :num_frames_down, ...].flatten(0,1)
            x = self.attn2(
                    norm_x, context=context, additional_tokens=additional_tokens
                )
            # Un merge Token first and then frame
            x = tu(x)
            x = x.reshape(batch_size, num_frames_down, seq_length * channels)
            x = fu(x).reshape(batch_frames, seq_length, channels)
            x = x + skip_x # Add skip connection to output of attn (self or cross) attn
            
            skip_x = x
            norm_x = self.norm3(x)

            # Frame merge for ff aka mlp
            fm, fu = fm_m, fu_m
            norm_x = norm_x.reshape(batch_size, num_frames, seq_length * channels)
            norm_x = fm(norm_x)
            num_frames_down = norm_x.shape[1]
            norm_x = norm_x.reshape(batch_size * num_frames_down, seq_length, channels)
            
            # Token merge for ff aka mlp
            tm, tu = tm_m, tu_m
            norm_x = tm(norm_x)
            
            x = self.ff(norm_x)
            
            # Un merge Token first and then frame
            x = tu(x)
            x = x.reshape(batch_size, num_frames_down, seq_length * channels)
            x = fu(x).reshape(batch_frames, seq_length, channels)
            x = x + skip_x # Add skip connection to output of ff (mlp)
            return x
    
    return ToMeBlock

def make_generative_models_tome_temp_block(block_class: Type[torch.nn.Module]) -> Type[torch.nn.Module]:
    """
    Make a patched class for a diffusers model.
    This patch applies ToMe to the forward function of the block.
    """
    class ToMeTempBlock(block_class):
        # Save for unpatching later
        _parent = block_class
        
        def _forward(self, x, context=None, timesteps=None):
            assert self.timesteps or timesteps
            assert not (self.timesteps and timesteps) or self.timesteps == timesteps
            timesteps = self.timesteps or timesteps
            B, S, C = x.shape
            b = B // timesteps
            
            x = rearrange(x, "(b t) s c -> b s (t c)", t=timesteps)
            # (1) ToMe
            m_i, m_a, m_c, m_m, u_i, u_a, u_c, u_m = compute_merge(x, self._tome_info, self._tome_info["args"]["bm_ratio"])
            x = rearrange(x, "b s (t c) -> (b t) s c", t=timesteps)
            x = rearrange(x, "(b t) s c -> (b s) t c", t=timesteps)

            if self.ff_in:
                x_skip = x
                x = self.norm_in(x)
                x = x.reshape(b, S, timesteps * C)
                # print("Calling m_i in temp_tome_block")
                # print("shape before ", x.shape)
                x = m_i(x)
                # print("shape after ", x.shape)
                seq_length_down = x.shape[1]
                x = x.reshape(b * seq_length_down, timesteps, C)
                
                x = self.ff_in(x)
                x = x.reshape(b, seq_length_down, timesteps * C)
                if self.is_res:
                    # print("Calling u_i in generative_models_tome_block")
                    x = u_i(x).reshape(b * S, timesteps, C) + x_skip

            x_skip = x.clone()
            x = self.norm1(x)
            
            if self.disable_self_attn:
                x = x.reshape(b, S, timesteps * C)
                # print("-"*50)
                # print("Calling m_c in after ff")
                # print("shape before ", x.shape)
                x = m_c(x)
                # print("shape after ", x.shape)
                # print("-"*50)
                seq_length_down = x.shape[1]
                x = x.reshape(b * seq_length_down, timesteps, C)
                x = self.attn1(x, context=context.reshape(S, b, 1, -1)[:seq_length_down, :, :, :].flatten(0,1))
                x = x.reshape(b, seq_length_down, timesteps * C)
                # print("Calling u_c in generative_models_tome_block after norm 1")
                x = u_c(x).reshape(b * S, timesteps, C) + x_skip
            else:
                x = x.reshape(b, S, timesteps * C)
                # print("-"*50)
                # print("Calling m_a in generative_models_tome_block after norm 1")
                # print("shape before ", x.shape)
                x = m_a(x) # size is (batch, seq_length_down, num_frames/timesteps * channels)
                # print("shape after ", x.shape)
                # print("-"*50)
                seq_length_down = x.shape[1]
                x = x.reshape(b * seq_length_down, timesteps, C)
                x = self.attn1(x)
                x = x.reshape(b, seq_length_down, timesteps * C)
                # print("Calling u_a in generative_models_tome_block after norm 1")
                x = u_a(x).reshape(b * S, timesteps, C) + x_skip

            x_skip = x.clone()
            x = self.norm2(x)
            if self.attn2 is not None:
                if self.switch_temporal_ca_to_sa:
                    x = x.reshape(b, S, timesteps * C)
                    # print("Calling m_a in generative_models_tome_block after norm2")
                    # print("shape before ", x.shape)
                    x = m_a(x) # size is (batch, seq_length_down, num_frames/timesteps * channels)
                    # print("shape after ", x.shape)
                    seq_length_down = x.shape[1]
                    x = x.reshape(b * seq_length_down, timesteps, C)
                    x = self.attn2(x)
                    x = x.reshape(b, seq_length_down, timesteps * C)
                    # print("Calling u_a in generative_models_tome_block")
                    x = u_a(x).reshape(b * S, timesteps, C) + x_skip
                else:
                    x = x.reshape(b, S, timesteps * C)
                    # print("Calling m_c in generative_models_tome_block after norm2")
                    # print("shape before ", x.shape)
                    x = m_c(x)
                    # print("shape after ", x.shape)
                    seq_length_down = x.shape[1]
                    x = x.reshape(b * seq_length_down, timesteps, C)
                    x = self.attn2(x, context=context.reshape(S, b, 1, -1)[:seq_length_down, :, :, :].flatten(0,1))
                    x = x.reshape(b, seq_length_down, timesteps * C)
                    # print("Calling u_c in generative_models_tome_block after norm2")
                    x = u_c(x).reshape(b * S, timesteps, C) + x_skip
            x_skip = x
            x = self.norm3(x)
            x = x.reshape(b, S, timesteps * C)
            # print("Calling m_m in generative_models_tome_block after norm3 [before last ff layer]")
            # print("shape before ", x.shape)
            x = m_m(x)
            # print("shape after ", x.shape)
            seq_length_down = x.shape[1]
            x = x.reshape(b * seq_length_down, timesteps, C)
            x = self.ff(x)
            x = x.reshape(b, seq_length_down, timesteps * C)
            if self.is_res:
                # print("Calling u_m in generative_models_tome_block after norm3 [after last ff layer]")
                x = u_m(x).reshape(b * S, timesteps, C)
                x = x + x_skip 

            x = x[None, :].reshape(b, S, timesteps, C)
            x = x.permute(0, 2, 1, 3)
            x = x.reshape(b * timesteps, S, C)
            # x = rearrange(
            #     x, "(b s) t c -> (b t) s c", s=S, b=B // timesteps, c=C, t=timesteps
            # )
            return x

    return ToMeTempBlock

def _chunked_feed_forward(
    ff: torch.nn.Module, hidden_states: torch.Tensor, chunk_dim: int, chunk_size: int
):
    # "feed_forward_chunk_size" can be used to save memory
    if hidden_states.shape[chunk_dim] % chunk_size != 0:
        raise ValueError(
            f"`hidden_states` dimension to be chunked: {hidden_states.shape[chunk_dim]} has to be divisible by chunk size: \
        {chunk_size}. Make sure to set an appropriate `chunk_size` when calling `unet.enable_forward_chunking`."
        )

    num_chunks = hidden_states.shape[chunk_dim] // chunk_size
    ff_output = torch.cat(
        [ff(hid_slice)
            for hid_slice in hidden_states.chunk(num_chunks, dim=chunk_dim)],
        dim=chunk_dim,
    )
    return ff_output

def make_diffusers_tome_block_sd3(
    block_class: Type[torch.nn.Module]
) -> Type[torch.nn.Module]:
    """
    Make a patched class for a diffusers model.
    This patch applies ToMe to the forward function of the block.
    """

    class ToMeBlock(block_class):
        # Save for unpatching later
        _parent = block_class

        def forward(
            self,
            hidden_states: torch.FloatTensor,
            encoder_hidden_states: torch.FloatTensor,
            temb: torch.FloatTensor,
        ) -> torch.Tensor:
            # print("Calling ToMeBlock (Dbolya) function")
            # == Custom code ==
            achieved_token_ratios = {}
            profiler_outputs = {}

            def measure_cuda_time(name, func, *args, **kwargs):
                # start = torch.cuda.Event(enable_timing=True)
                # end = torch.cuda.Event(enable_timing=True)

                # torch.cuda.synchronize()
                # start.record()
                result = func(*args, **kwargs)
                # end.record()
                # torch.cuda.synchronize()
                # cuda_time = start.elapsed_time(end)
                # profiler_outputs[name] = {"cuda_time": cuda_time}
                return result

            ## ======================== ##

            norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                measure_cuda_time("norm1", self.norm1, hidden_states, emb=temb)
            )

            if self.context_pre_only:
                norm_encoder_hidden_states = measure_cuda_time(
                    "norm1_context", self.norm1_context, encoder_hidden_states, temb
                )
            else:
                (
                    norm_encoder_hidden_states,
                    c_gate_msa,
                    c_shift_mlp,
                    c_scale_mlp,
                    c_gate_mlp,
                ) = measure_cuda_time(
                    "norm1_context", self.norm1_context, encoder_hidden_states, emb=temb
                )

            m_a, _, m_ff, u_a, _, u_ff = compute_merge(
                x=norm_hidden_states,
                tome_info=self._tome_info,)
            
            # [Patch] Apply hierarchical merge
            # _prev_tokens = norm_hidden_states.shape[1]
            # print("# of tokens before doing attn compression", _prev_tokens)
            # print(f"Tensor shape before compression: {norm_hidden_states.shape}")
            norm_hidden_states = measure_cuda_time(
                "merge_attn", m_a, norm_hidden_states
            )
            # _next_tokens = norm_hidden_states.shape[1]
            # print("# of tokens after doing attn compression", _next_tokens)

            # print(f"# of tokens after compression: {_next_tokens}")
            # print(f"Total Compression Ratio: {_next_tokens/_prev_tokens * 100:.2f}%")

            # print(f"Token merge ratio achieved (attn) {1 - (_next_tokens / _prev_tokens)}")
            # If achieved token ration is not set for joint attn, then set it
            # if "joint_attn" not in achieved_token_ratios:
            #     achieved_token_ratios["joint_attn"] = 1 - \
            #         (_next_tokens / _prev_tokens)

            # Attention.
            attn_output, context_attn_output = measure_cuda_time(
                "joint_self_attn",
                self.attn,
                hidden_states=norm_hidden_states,
                encoder_hidden_states=norm_encoder_hidden_states,
            )
            
            # [Patch] Apply hierarchical unmerge
            attn_output = measure_cuda_time(
                "unmerge_attn",
                u_a,
                attn_output,
            )

            # Process attention outputs for the `hidden_states`.
            attn_output = gate_msa.unsqueeze(1) * attn_output
            hidden_states = hidden_states + attn_output

            norm_hidden_states = measure_cuda_time(
                "norm2_before_mlp", self.norm2, hidden_states
            )
            norm_hidden_states = (
                norm_hidden_states *
                (1 + scale_mlp[:, None]) + shift_mlp[:, None]
            )

            # [Patch] Compute hierarchical merge for MLP layer
            if self._chunk_size is not None:
                # == Add custom code (because of chunked feed-forward) ==
                # torch.cuda.synchronize()
                # start.record()

                # [Patch] Apply hierarchical merge
                # _prev_tokens = norm_hidden_states.shape[1]
                # print("# of Tokens before doing mlp compression (hidden_states)", _prev_tokens)
                norm_hidden_states = measure_cuda_time(
                    "merge_mlp_hidden_states", m_ff, norm_hidden_states
                )
                # _next_tokens = norm_hidden_states.shape[1]
                # print("# of Tokens after doing mlp compression (hidden_states)", _next_tokens)

                
                # print(f"Token merge ratio achieved (ffn) {1 - (_next_tokens / _prev_tokens)}")
                # if "mlp_hidden_states" not in achieved_token_ratios:
                #     achieved_token_ratios["mlp_hidden_states"] = 1 - (
                #         _next_tokens / _prev_tokens
                #     )

                # "feed_forward_chunk_size" can be used to save memory
                ff_output = _chunked_feed_forward(
                    self.ff, norm_hidden_states, self._chunk_dim, self._chunk_size
                )
                
        
                ff_output = measure_cuda_time(
                    "unmerge_mlp_hidden_states",
                    u_ff,
                    ff_output,
                )

                # torch.cuda.synchronize()
                # end.record()
                # cuda_time = start.elapsed_time(end)
                # profiler_outputs["mlp_hidden_states"] = {
                #     "cuda_time": cuda_time}
            else:
                norm_hidden_states = measure_cuda_time(
                    "merge_mlp_hidden_states", m_ff, norm_hidden_states
                )
                # _next_tokens = norm_hidden_states.shape[1]
                # print(f"# of tokens after mlp compression (hidden_states): {norm_hidden_states.shape[1]}")

                # print(f"Token merge ratio achieved (ffn) {1 - (_next_tokens / _prev_tokens)}")
                # if "mlp_hidden_states" not in achieved_token_ratios:
                #     achieved_token_ratios["mlp_hidden_states"] = 1 - (
                #         _next_tokens / _prev_tokens
                #     )

                ff_output = measure_cuda_time(
                    "mlp_hidden_states", self.ff, norm_hidden_states
                )

                ff_output = measure_cuda_time(
                    "unmerge_mlp_hidden_states",
                    u_ff,
                    ff_output,
                )

            ff_output = gate_mlp.unsqueeze(1) * ff_output

            hidden_states = hidden_states + ff_output

            # Process attention outputs for the `encoder_hidden_states`.
            if self.context_pre_only:
                encoder_hidden_states = None
            else:
                context_attn_output = c_gate_msa.unsqueeze(
                    1) * context_attn_output
                encoder_hidden_states = encoder_hidden_states + context_attn_output

                norm_encoder_hidden_states = measure_cuda_time(
                    "norm2_context", self.norm2_context, encoder_hidden_states
                )
                norm_encoder_hidden_states = (
                    norm_encoder_hidden_states * (1 + c_scale_mlp[:, None])
                    + c_shift_mlp[:, None]
                )
                if self._chunk_size is not None:
                    # == Add custom code (because of chunked feed-forward) ==
                    # torch.cuda.synchronize()
                    # start.record()

                    # "feed_forward_chunk_size" can be used to save memory
                    context_ff_output = _chunked_feed_forward(
                        self.ff_context,
                        norm_encoder_hidden_states,
                        self._chunk_dim,
                        self._chunk_size,
                    )

                    # torch.cuda.synchronize()
                    # end.record()
                    # cuda_time = start.elapsed_time(end)
                    # profiler_outputs["mlp_encoder_states"] = {
                    #     "cuda_time": cuda_time}
                else:
                    context_ff_output = measure_cuda_time(
                        "mlp_encoder_states",
                        self.ff_context,
                        norm_encoder_hidden_states,
                    )
                encoder_hidden_states = (
                    encoder_hidden_states +
                    c_gate_mlp.unsqueeze(1) * context_ff_output
                )

            # self._profiler_outputs = profiler_outputs
            # self.achieved_token_ratios = achieved_token_ratios

            return encoder_hidden_states, hidden_states

    return ToMeBlock


def hook_tome_model(model: torch.nn.Module):
    """ Adds a forward pre hook to get the image size. This hook can be removed with remove_patch. """
    def hook(module, args):
        module._tome_info["size"] = (args[0].shape[2], args[0].shape[3])
        return None

    model._tome_info["hooks"].append(model.register_forward_pre_hook(hook))








def apply_patch(
        model: torch.nn.Module,
        ratio: float = 0.5,
        # tm_ratio_mlp: float = 0.5,
        # bm_ratio: float = 0.5,
        max_downsample: int = 1,
        sx: int = 2,
        sy: int = 2,
        dst_tokens_per_stride: int = 1,
        # st: int = 2,
        use_rand: bool = True,
        merge_attn: bool = True,
        merge_crossattn: bool = True,
        merge_mlp: bool = True,
        use_triton: bool = False,
        ):
    """
    Patches a stable diffusion model with ToMe.
    Apply this to the highest level stable diffusion object (i.e., it should have a .model.diffusion_model).

    Important Args:
     - model: A top level Stable Diffusion module to patch in place. Should have a ".model.diffusion_model"
     - ratio: The ratio of tokens to merge. I.e., 0.4 would reduce the total number of tokens by 40%.
              The maximum value for this is 1-(1/(sx*sy)). By default, the max is 0.75 (I recommend <= 0.5 though).
              Higher values result in more speed-up, but with more visual quality loss.
    
    Args to tinker with if you want:
     - max_downsample [1, 2, 4, or 8]: Apply ToMe to layers with at most this amount of downsampling.
                                       E.g., 1 only applies to layers with no downsampling (4/15) while
                                       8 applies to all layers (15/15). I recommend a value of 1 or 2.
     - sx, sy: The stride for computing dst sets (see paper). A higher stride means you can merge more tokens,
               but the default of (2, 2) works well in most cases. Doesn't have to divide image size.
     - use_rand: Whether or not to allow random perturbations when computing dst sets (see paper). Usually
                 you'd want to leave this on, but if you're having weird artifacts try turning this off.
     - merge_attn: Whether or not to merge tokens for attention (recommended).
     - merge_crossattn: Whether or not to merge tokens for cross attention (not recommended).
     - merge_mlp: Whether or not to merge tokens for the mlp layers (very not recommended).
    """

    # print(f"fm ratio:{fm_ratio}, tm ratio:{tm_ratio}, bm_ratio:{bm_ratio}")
    # Make sure the module is not currently patched
    remove_patch(model)

    is_diffusers = isinstance_str(model, "DiffusionPipeline") or isinstance_str(model, "ModelMixin")
    
    is_openai_wrapper = hasattr(model, "model") and isinstance_str(model.model, "OpenAIWrapper")

    if not is_diffusers and not is_openai_wrapper:
        if not hasattr(model, "model") or not hasattr(model.model, "diffusion_model"):
            # Provided model not supported
            raise RuntimeError("Provided model was not a Stable Diffusion / Latent Diffusion model, as expected.")
        diffusion_model = model.model.diffusion_model
    elif is_openai_wrapper:
        diffusion_model = model.model.diffusion_model # Patch for code that builds on StabilityAI - generative models repo
    else:
        # Supports "pipe.unet" and "unet"
        # diffusion_model = model.unet if hasattr(model, "unet") else model
        if hasattr(model, "unet"):
            diffusion_model = model.unet
        elif hasattr(model, "transformer"):
            diffusion_model = model.transformer
        else:
            diffusion_model = model

    diffusion_model._tome_info = {
        "size": (128 // 2, 128 // 2),  # Hardcoded patch size values
        "num_frames": 1,
        "hooks": [],
        "args": {
            # "fm_ratio": fm_ratio,  # frame merge ratio
            "ratio": ratio,  # token merge ratio for joint attn layers
            # "tm_ratio_mlp": tm_ratio_mlp,  # token merge ratio for mlp layers
            # "bm_ratio": bm_ratio,  # block merge ratio
            "max_downsample": max_downsample,
            "sx": sx,
            "sy": sy,
            # "st": st,
            "dst_tokens_per_stride": dst_tokens_per_stride,
            "use_rand": use_rand,
            "generator": None,
            "merge_attn": merge_attn,
            "merge_crossattn": merge_crossattn,
            "merge_mlp": merge_mlp,
            "use_triton": use_triton,
        },
    }
    # hook_tome_model(diffusion_model)
    
    # exclude_names = [
    #     "input_blocks.1.1.transformer_blocks",
    #     "input_blocks.2.1.transformer_blocks",
    #     "input_blocks.4.1.transformer_blocks",
    #     "input_blocks.5.1.transformer_blocks",
    #     "input_blocks.7.1.transformer_blocks",
    #     "input_blocks.8.1.transformer_blocks",
    #     "middle_block.1.transformer_blocks",
    #     "output_blocks.3.1.transformer_blocks",
    #     "output_blocks.4.1.transformer_blocks",
    #     "output_blocks.5.1.transformer_blocks",
    #     "output_blocks.6.1.transformer_blocks",
    #     "output_blocks.7.1.transformer_blocks",
    #     "output_blocks.8.1.transformer_blocks",
    #     "output_blocks.9.1.transformer_blocks",
    #     "output_blocks.10.1.transformer_blocks",
    #     "output_blocks.11.1.transformer_blocks",
    # ]
    
    include_names_gp1 = [
        "transformer_blocks.0",
        "transformer_blocks.1",
        "transformer_blocks.2",
        "transformer_blocks.3",
        "transformer_blocks.4",
        "transformer_blocks.5",
        "transformer_blocks.6",
        "transformer_blocks.7",
        "transformer_blocks.8",
        "transformer_blocks.9",
        "transformer_blocks.10",
        "transformer_blocks.11",
        "transformer_blocks.12",
        "transformer_blocks.13",
        "transformer_blocks.14",
        "transformer_blocks.15",
        "transformer_blocks.16",
        "transformer_blocks.17",
        "transformer_blocks.18",
        "transformer_blocks.19",
        "transformer_blocks.20",
        "transformer_blocks.21",
        "transformer_blocks.22",
        "transformer_blocks.23",
        ]

    for name, module in diffusion_model.named_modules():
        # If for some reason this has a different name, create an issue and I'll fix it
        # if name contains any of the exclude_names, skip
        if isinstance_str(module, "JointTransformerBlock"):
            if any([include_name == name for include_name in list(include_names_gp1)]):
                print(f"Applying ToMe to {name}")
                make_tome_block_fn = make_diffusers_tome_block_sd3
                module.__class__ = make_tome_block_fn(module.__class__)
                module._tome_info = copy.deepcopy(diffusion_model._tome_info)
                print(
                    f"""Patching {name} with  tm_ratio_attn: {module._tome_info['args']['ratio']} and tm_ratio_mlp: {
                        module._tome_info['args']['ratio']}"""
                )
                # Something introduced in SD 2.0 (LDM only)
                if not hasattr(module, "disable_self_attn") and not is_diffusers:
                    module.disable_self_attn = False

                # Something needed for older versions of diffusers
                if not hasattr(module, "use_ada_layer_norm_zero") and is_diffusers:
                    module.use_ada_layer_norm = False
                    module.use_ada_layer_norm_zero = False

    print("Done patching")
    return model





def remove_patch(model: torch.nn.Module):
    """ Removes a patch from a ToMe Diffusion module if it was already patched. """
    # For diffusers
    model = model.unet if hasattr(model, "unet") else model

    for _, module in model.named_modules():
        if hasattr(module, "_tome_info"):
            for hook in module._tome_info["hooks"]:
                hook.remove()
            module._tome_info["hooks"].clear()

        if module.__class__.__name__ == "ToMeBlock":
            module.__class__ = module._parent
    
    return model
