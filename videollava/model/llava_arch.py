#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import torch.nn.functional as F
from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from .multimodal_encoder.builder import build_image_tower, build_video_tower
from .multimodal_projector.builder import build_vision_projector

from videollava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
import time

class LlavaMetaModel:

    def __init__(self, config):
        super(LlavaMetaModel, self).__init__(config)

        if getattr(config, "mm_image_tower", None) is not None:
            self.image_tower = build_image_tower(config, delay_load=True)
        if getattr(config, "mm_video_tower", None) is not None:
            self.video_tower = build_video_tower(config, delay_load=True)
        if getattr(config, "mm_image_tower", None) is not None or getattr(config, "mm_video_tower", None) is not None:
            self.mm_projector = build_vision_projector(config)

    def get_image_tower(self):
        image_tower = getattr(self, 'image_tower', None)
        if type(image_tower) is list:
            image_tower = image_tower[0]
        return image_tower

    def get_video_tower(self):
        video_tower = getattr(self, 'video_tower', None)
        if type(video_tower) is list:
            video_tower = video_tower[0]
        return video_tower

    def initialize_vision_modules(self, model_args, fsdp=None):
        # ==============================================
        image_tower = model_args.image_tower
        video_tower = model_args.video_tower
        assert image_tower is not None or video_tower is not None
        # ==============================================
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter

        # ==========================================================================

        self.config.mm_image_tower = image_tower
        if image_tower is not None:
            if self.get_image_tower() is None:
                image_tower = build_image_tower(model_args)

                if fsdp is not None and len(fsdp) > 0:
                    self.image_tower = [image_tower]
                else:
                    self.image_tower = image_tower
            else:
                if fsdp is not None and len(fsdp) > 0:
                    image_tower = self.image_tower[0]
                else:
                    image_tower = self.image_tower
                image_tower.load_model()

        self.config.mm_video_tower = video_tower
        if video_tower is not None:
            if self.get_video_tower() is None:
                video_tower = build_video_tower(model_args)

                if fsdp is not None and len(fsdp) > 0:
                    self.video_tower = [video_tower]
                else:
                    self.video_tower = video_tower
            else:
                if fsdp is not None and len(fsdp) > 0:
                    video_tower = self.video_tower[0]
                else:
                    video_tower = self.video_tower
                video_tower.load_model()

        # ==========================================================================

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature
        # ==========================================================================
        if image_tower is not None and video_tower is not None:  # TODO: support different hidden_size
            assert image_tower.hidden_size == video_tower.hidden_size
            self.config.mm_hidden_size = image_tower.hidden_size
        else:
            self.config.mm_hidden_size = max(getattr(image_tower, 'hidden_size', -1),
                                             getattr(video_tower, 'hidden_size', -1))
        # ===================================================================================

        if getattr(self, 'mm_projector', None) is None:
            self.mm_projector = build_vision_projector(self.config)
        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))


class LlavaMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_image_tower(self):
        return self.get_model().get_image_tower()

    def get_video_tower(self):
        return self.get_model().get_video_tower()

    def encode_images(self, images):
        image_features = self.get_model().get_image_tower()(images)
        image_features = self.get_model().mm_projector(image_features)
      
        return image_features

    def diff_temporal_merging(self, video_features, tomerge_size):  # [1, 8, 257, 4096]
        bsz, t, pn, hd = video_features.shape
        diff_video_feature = video_features[:, 1:, :, :] - video_features[:, :-1, :, :]  # [1, 7, 257, 4096]
        diff = torch.mean(torch.abs(diff_video_feature), dim=-1)  # [1, 7, 257]
        diff = torch.sum(diff, dim=-1) # [1, 7]
        _, idx_tomerge = torch.topk(-diff, k=tomerge_size, dim=-1)
        idx_tomerge, _ = torch.sort(idx_tomerge[0])

        merge_mask = torch.zeros((t - tomerge_size, t), dtype=torch.bool)
        last_idx = -2
        row = -1
        for idx in idx_tomerge:
            if idx - last_idx == 1:
                merge_mask[row][idx:idx + 2] = True
            elif idx - last_idx == 2:
                row += 1
                merge_mask[row][idx:idx + 2] = True
            else:
                for i in range(last_idx + 2, idx):
                    row += 1
                    merge_mask[row, i] = True
                row += 1
                merge_mask[row][idx:idx + 2] = True
            last_idx = idx
        if row != (t - tomerge_size - 1):
            for i in range(last_idx + 2, t):
                row += 1
                merge_mask[row, i] = True
        # merge_mask: [t-tomerge_size, 8] 选出待merge的tokens
        merge_mask = merge_mask.view(1, t - tomerge_size, t, 1, 1).repeat(bsz, 1, 1, pn, hd).to(video_features.device)
        merged_video_features = (torch.sum(merge_mask * video_features.unsqueeze(1), dim=2)
                                 / torch.sum(merge_mask, dim=2))

        return merged_video_features

    def sim_temporal_merging(self, video_features, tomerge_size):  # [1, 8, 257, 4096]
        bsz, t, pn, hd = video_features.shape

        norm = torch.norm(video_features, dim=-1).unsqueeze(-1)
        video_features0 = video_features[:, :-1, :, :] / norm[:, :-1, :, :] # [1, 7, 257, 4096]
        video_features1 = video_features[:, 1:, :, :] / norm[:, 1:, :, :]   # [1, 7, 257, 4096]
        sim = (video_features0.unsqueeze(-2) @ video_features0.unsqueeze(-1))[..., 0, 0] # [1, 7, 257, 1, 1]
        sim = torch.sum(sim, dim=2)
        _, idx_tomerge = torch.topk(sim, k=tomerge_size, dim=-1)
        idx_tomerge, _ = torch.sort(idx_tomerge[0])

        merge_mask = torch.zeros((t - tomerge_size, t), dtype=torch.bool)
        last_idx = -2
        row = -1
        for idx in idx_tomerge:
            if idx - last_idx == 1:
                merge_mask[row][idx:idx + 2] = True
            elif idx - last_idx == 2:
                row += 1
                merge_mask[row][idx:idx + 2] = True
            else:
                for i in range(last_idx + 2, idx):
                    row += 1
                    merge_mask[row, i] = True
                row += 1
                merge_mask[row][idx:idx + 2] = True
            last_idx = idx
        if row != (t - tomerge_size - 1):
            for i in range(last_idx + 2, t):
                row += 1
                merge_mask[row, i] = True
        # merge_mask: [t-tomerge_size, 8] 选出待merge的tokens
        merge_mask = merge_mask.view(1, t - tomerge_size, t, 1, 1).repeat(bsz, 1, 1, pn, hd).to(video_features.device)
        merged_video_features = (torch.sum(merge_mask * video_features.unsqueeze(1), dim=2)
                                 / torch.sum(merge_mask, dim=2))

        return merged_video_features

    def spatial_pruning(self, video_features, keep_ratio):
        bsz, t, pn, hd = video_features.shape
        keep_size = int(pn * keep_ratio)

        norm = torch.norm(video_features, dim=-1).unsqueeze(-1)

        cls = video_features[:, :, 0:1, :]  # [1, 7, 1, 4096]
        tokens = video_features[:, :, 1:, :]  # [1, 7, 256, 4096]
        sim = (tokens / norm[:, :, 1:, :]) @ (cls / norm[:, :, 0:1, :]).transpose(2, 3)  # [1, 7, 256, 1]
        _, keep_topk = torch.topk(sim, k=keep_size - 1, dim=-2)
        keep_topk = keep_topk.sort().values.repeat(bsz, 1, 1, hd)
        mask = torch.zeros(tokens.shape, dtype=torch.bool).to(tokens.device)
        mask = mask.scatter(-2, keep_topk, 1)
        tokens = tokens[mask]
        tokens = tokens.view(bsz, t, keep_size - 1, hd)
        keep_video_feature = torch.cat((cls, tokens), dim=-2)

        return keep_video_feature

    def att_compr(self,A, B, K=0.6):
        # 假设A和B是已经定义的张量
        # A的尺寸: (1, 8, 257, 4096)
        # B的尺寸: (54, 4096)
        # K: 压缩比例 (0.2 表示压缩到原来的20%)

        # 计算A和B的点积来衡量相似性
        A_flattened = A.view(1, 8, 257, 1, 4096)  # (1, 8, 257, 1, 4096)
        B_expanded = B.unsqueeze(0).unsqueeze(0).unsqueeze(2)  # (1, 1, 1, 54, 4096)
        similarity = torch.sum(A_flattened * B_expanded, dim=-1)  # (1, 8, 257, 54)

        # 计算注意力权重，使用Softmax进行归一化
        attention_weights = F.softmax(similarity, dim=2)  # (1, 8, 257, 54)

        # 计算注意力权重的均值，得到每个通道的整体注意力权重
        average_attention = attention_weights.mean(dim=3)  # (1, 8, 257)

        # 找到最相关的部分，根据K值选择前K%的部分
        num_to_keep = int(A.size(2) * K)
        _, indices = torch.topk(average_attention, num_to_keep, dim=2)  # (1, 8, num_to_keep)

        # 使用选中的索引压缩A
        indices = indices.unsqueeze(-1).expand(-1, -1, -1, A.size(-1))  # (1, 8, num_to_keep, 4096)
        A_compressed = torch.gather(A, 2, indices)  # (1, 8, num_to_keep, 4096)

        return A_compressed
    
    def cos_compr(self,A, B, K=0.6):
        # A的尺寸: (1, 8, 257, 4096)
        # B的尺寸: (54, 4096)
        # K: 压缩比例 (0.2 表示压缩到原来的20%)

        # 计算余弦相似度
        A_flattened = A.view(1, 8, 257, 1, 4096)  # (1, 8, 257, 1, 4096)
        B_expanded = B.unsqueeze(0).unsqueeze(0).unsqueeze(2)  # (1, 1, 1, 54, 4096)
        cos_sim = F.cosine_similarity(A_flattened, B_expanded, dim=-1)  # (1, 8, 257, 54)

        # 计算相似度的均值，得到每个通道的整体相似度
        average_sim = cos_sim.mean(dim=3)  # (1, 8, 257)

        # 找到最相关的部分，根据K值选择前K%的部分
        num_to_keep = int(A.size(2) * K)
        _, indices = torch.topk(average_sim, num_to_keep, dim=2)  # (1, 8, num_to_keep)

        # 使用选中的索引压缩A
        indices = indices.unsqueeze(-1).expand(-1, -1, -1, A.size(-1))  # (1, 8, num_to_keep, 4096)
        A_compressed = torch.gather(A, 2, indices)  # (1, 8, num_to_keep, 4096)

        return A_compressed
    
    def pool_compr(self,A, B, K=0.6):
        # 假设A和B是已经定义的张量
        # A的尺寸: (1, 8, 257, 4096)
        # B的尺寸: (54, 4096)
        # K: 压缩比例 (0.2 表示压缩到原来的20%)

        # 计算A和B的相似度
        A_flattened = A.view(1, 8, 257, 1, 4096)  # (1, 8, 257, 1, 4096)
        B_expanded = B.unsqueeze(0).unsqueeze(0).unsqueeze(2)  # (1, 1, 1, 54, 4096)
        similarity = torch.sum(A_flattened * B_expanded, dim=-1)  # (1, 8, 257, 54)

        # 计算相似度的均值，得到每个通道的整体相似度
        average_similarity = similarity.mean(dim=3)  # (1, 8, 257)

        # 找到最相关的部分，根据K值选择前K%的部分
        num_to_keep = int(A.size(2) * K)
        _, indices = torch.topk(average_similarity, num_to_keep, dim=2)  # (1, 8, num_to_keep)

        # 使用选中的索引对A进行池化操作
        indices = indices.unsqueeze(-1).expand(-1, -1, -1, A.size(-1))  # (1, 8, num_to_keep, 4096)
        A_pooled = torch.gather(A, 2, indices)  # (1, 8, num_to_keep, 4096)

        return A_pooled
    
    
        
    def encode_videos(self, videos, num_images,text_embeds):  # [mini_b, c, t, h, w]
        # [1,3,8,224,224]
        # [1,8,257,1024]
        # [1,8,257,4096]
        b, _, t, _, _ = videos.shape
        
        # t1 = time.time()
        video_features = self.get_model().get_video_tower()(videos)  # [mini_b, t, n, c]
        # t2 = time.time()
        
        video_features = self.get_model().mm_projector(video_features)
        
        # t3 = time.time()
        video_features = self.spatial_pruning(video_features, keep_ratio=0.25)
        # t4 = time.time()
        # video_features = self.diff_temporal_merging(video_features, tomerge_size=t-num_images)
        video_features = self.sim_temporal_merging(video_features, tomerge_size=t-num_images)
        # t5 = time.time()
        # print(t5-t3)
        
        
        
  
        
     
        
        
        return video_features[:,:, :, :]

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels, images
    ):
        # ====================================================================================================
        image_tower = self.get_image_tower()
        video_tower = self.get_video_tower()
        if (image_tower is None and video_tower is None) or images is None or input_ids.shape[1] == 1:
            if past_key_values is not None and (image_tower is not None or video_tower is not None) and images is not None and input_ids.shape[1] == 1:
                target_shape = past_key_values[-1][-1].shape[-2] + 1
                attention_mask = torch.cat((attention_mask, torch.ones(
                    (attention_mask.shape[0], target_shape - attention_mask.shape[1]),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device
                )), dim=1)
                position_ids = torch.sum(attention_mask, dim=1).unsqueeze(-1) - 1
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        '''
            images is a list, if batch_size=6
            [
                image(3, 224, 224),      # sample 1
                image(3, 224, 224),      # sample 2
                video(t, 3, 224, 224),   # sample 3
                image(3, 224, 224),      # sample 4
                image(3, 224, 224),      # sample 4
                video(t, 3, 224, 224),   # sample 5
                video(t, 3, 224, 224),   # sample 5
                video(t, 3, 224, 224),   # sample 6
                image(3, 224, 224),      # sample 6
            ]
            will be converted to image_features, all video_feature will be flatten as image
            [
                [n, c],                  # sample 1
                [n, c),                  # sample 2
                *(t * [new_n, c]),       # sample 3
                [n, c],                  # sample 4
                [n, c],                  # sample 4
                *(t * [new_n, c]),       # sample 5
                *(t * [new_n, c]),       # sample 5
                *(t * [new_n, c]),       # sample 6
                [n, c],                  # sample 6
            ]
        '''
        image_idx = [idx for idx, img in enumerate(images) if img.ndim == 3]
        is_all_image = len(image_idx) == len(images)
        video_idx = [idx for idx, vid in enumerate(images) if vid.ndim == 4]
        images_minibatch = torch.stack([images[idx] for idx in image_idx]) if len(image_idx) > 0 else []  # mini_b c h w
        videos_minibatch = torch.stack([images[idx] for idx in video_idx]) if len(video_idx) > 0 else []  # mini_b c t h w
        
        
        # print(len(image_features), *[i.shape for i in image_features])
        # print(len(image_features), image_features[0].shape)
        # ====================================================================================================

        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
            raise NotImplementedError

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- TODO: double check
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        
        cur_input_ids = input_ids[0]
        num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
        

        image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
        cur_input_ids_noim = []
        cur_labels = labels[0]
        cur_labels_noim = []
        for i in range(len(image_token_indices) - 1):
            cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]])
            cur_labels_noim.append(cur_labels[image_token_indices[i]+1:image_token_indices[i+1]])
        split_sizes = [x.shape[0] for x in cur_labels_noim]
        
        cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
        
        cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
        
        ################# question-based compression ######################
        
        
        tmp_image_features = [None] * (len(image_idx) + len(video_idx))
        if getattr(images_minibatch, 'ndim', 0) == 4:  # batch consists of images, [mini_b, c, h, w]
            if image_tower is not None:
                image_features_minibatch = self.encode_images(images_minibatch)  # [mini_b, l, c]
            else:
                image_features_minibatch = torch.randn(1).to(self.device)  # dummy feature for video-only training under tuning
            for i, pos in enumerate(image_idx):
                tmp_image_features[pos] = image_features_minibatch[i]

        if getattr(videos_minibatch, 'ndim', 0) == 5:  # batch consists of videos, [mini_b, c, t, h, w]
            num_images = (input_ids[0] == IMAGE_TOKEN_INDEX).sum()
            
            video_features_minibatch = self.encode_videos(videos_minibatch, num_images,cur_input_embeds)  # fake list [mini_b, t, l, c]
            # t1 = time.time()
            # video_features_minibatch = self.pool_compr(video_features_minibatch,cur_input_embeds,K=0.5)
            # t2 = time.time()
            # print(t2-t1)
            # print(f'video_features_minibatch: {video_features_minibatch.shape}') # [1, 8, 257, 4096]
            for i, pos in enumerate(video_idx):
                t = video_features_minibatch[i].shape[0]
                tmp_image_features[pos] = [video_features_minibatch[i][j] for j in range(t)]

        new_tmp = []
        for image in tmp_image_features:
            # print(len(new_tmp), len(image))
            if isinstance(image, list):
                t = len(image)
                for i in range(t):
                    new_tmp.append(image[i])
                # print('add video')
            else:
                new_tmp.append(image)
        image_features = new_tmp
        # print(image_features.shape)
        ###################################################################
        
        cur_new_input_embeds = []
        cur_new_labels = []

        for i in range(num_images + 1):
            cur_new_input_embeds.append(cur_input_embeds_no_im[i])
            cur_new_labels.append(cur_labels_noim[i])
            if i < num_images:
                # print(cur_image_idx)
                cur_image_features = image_features[cur_image_idx]
                cur_image_idx += 1
                cur_new_input_embeds.append(cur_image_features)
                cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

        cur_new_input_embeds = torch.cat(cur_new_input_embeds)  #change it to [2110,4096]   8*257+35+19
        cur_new_labels = torch.cat(cur_new_labels)

        new_input_embeds.append(cur_new_input_embeds)
        new_labels.append(cur_new_labels)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False
