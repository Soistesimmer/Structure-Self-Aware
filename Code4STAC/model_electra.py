import torch
from torch import nn

from dialogue_dataset import DialogueDataset


class StudentModel(nn.Module):
    def __init__(self, params, pretrained_model):
        super().__init__()
        self.pretrained_model = pretrained_model
        self.dialog_gru = nn.GRU(params.hidden_size, params.hidden_size // 2, batch_first=True, bidirectional=True)

        self.path_emb = PathEmbedding(params)
        self.path_update = PathUpdateModel(params)
        self.gnn = StructureAwareAttention(params.hidden_size, params.path_hidden_size, params.num_heads, params.dropout)

        self.link_classifier = Classifier(params.hidden_size * 4 + params.path_hidden_size, params.hidden_size, 1)
        self.label_classifier = Classifier(params.hidden_size * 4 + params.path_hidden_size, params.hidden_size,
                                           params.relation_type_num)
        self.layer_num = params.num_layers
        self.norm = nn.LayerNorm(params.hidden_size)
        self.dropout = nn.Dropout(params.dropout)
        self.root = nn.Parameter(torch.zeros(params.hidden_size), requires_grad=False)


    def forward(self, texts, lengths, edu_nums, speakers, turns):
        batch_size, edu_num, sentence_len = texts.size()
        node_num = edu_num + 1

        texts = texts.reshape(batch_size * edu_num, sentence_len)
        attn_mask = DialogueDataset.padding(texts, padding_value=0)
        sentences = self.pretrained_model(texts, attention_mask=attn_mask)[0][:, 0, :]
        sentences = self.dropout(sentences)

        dialog_input = torch.cat((self.root.expand(batch_size, 1, sentences.size(-1)),
                                  sentences.reshape(batch_size, edu_num, -1)),dim=1)

        dialog_output, dialog_hx = self.dialog_gru(dialog_input)
        dialog_output = self.dropout(dialog_output)

        edu_nums = edu_nums + 1
        edu_attn_mask = torch.arange(edu_nums.max()).expand(len(edu_nums), edu_nums.max()).cuda() < edu_nums.unsqueeze(
            1)
        edu_attn_mask = self.gnn.masking_bias(edu_attn_mask)

        nodes = dialog_output
        const_path = self.path_emb(speakers, turns)
        memory = []
        struct_path=torch.zeros_like(const_path)
        struct_update_mask = self.get_update_mask(struct_path)
        for _ in range(self.layer_num):
            nodes = self.gnn(nodes, edu_attn_mask, self.dns2ds(struct_path)+const_path)
            struct_path = self.path_update(nodes, const_path, struct_path, struct_update_mask)
            memory.append(struct_path)
            struct_path=self.dropout(struct_path)

        predicted_path = struct_path

        multi_agent = torch.cat((dialog_input, dialog_output), dim=-1)
        expand_multi_agent = multi_agent.unsqueeze(1).expand(batch_size, node_num, node_num, multi_agent.size(-1))
        multi_actors = torch.cat((expand_multi_agent, expand_multi_agent.transpose(1, 2), predicted_path), dim=-1)

        return self.link_classifier(multi_actors).squeeze(-1), \
               self.label_classifier(multi_actors), memory

    def get_update_mask(self, path):
        batch_size, node_num, _, _ = path.size()
        mask = torch.tril(torch.ones(node_num, node_num).bool(), -1).expand(batch_size, node_num, node_num)
        return mask

    @staticmethod
    def dns2ds(path):
        batch_size, node_num, _, _ = path.size()
        mask = torch.triu(torch.ones(node_num, node_num).bool(), 1).expand(batch_size, node_num, node_num)
        path[mask] = path.transpose(1, 2)[mask]
        return path


class TeacherModel(nn.Module):
    def __init__(self, params, pretrained_model):
        super().__init__()
        self.pretrained_model = pretrained_model
        self.dialog_gru = nn.GRU(params.hidden_size, params.hidden_size // 2, batch_first=True, bidirectional=True)

        self.path_emb = PathEmbedding(params)
        self.path_model = PathModel(params)
        self.path_update = PathUpdateModel(params)
        self.layer_num = params.num_layers
        self.gnn = StructureAwareAttention(params.hidden_size, params.path_hidden_size, params.num_heads,
                                           params.dropout)

        self.link_classifier = Classifier(params.hidden_size * 4 + params.path_hidden_size, params.hidden_size, 1)
        self.label_classifier = Classifier(params.hidden_size * 4 + params.path_hidden_size, params.hidden_size,
                                           params.relation_type_num)

        self.norm = nn.LayerNorm(params.hidden_size)
        self.dropout = nn.Dropout(params.dropout)
        self.root = nn.Parameter(torch.zeros(params.hidden_size), requires_grad=False)

        self.hidden_size = params.hidden_size
        self.path_hidden_size = params.path_hidden_size

    def forward(self, texts, lengths, edu_nums, speakers, turns, graphs):
        batch_size, edu_num, sentence_len = texts.size()
        node_num = edu_num + 1

        texts = texts.reshape(batch_size * edu_num, sentence_len)
        attn_mask = DialogueDataset.padding(texts, padding_value=0)
        sentences = self.pretrained_model(texts, attention_mask=attn_mask)[0][:, 0, :]

        dialog_input = torch.cat((self.root.expand(batch_size, 1, sentences.size(-1)),
                                  sentences.reshape(batch_size, edu_num, -1)),dim=1) # B N H

        dialog_output, dialog_hx = self.dialog_gru(dialog_input)
        dialog_output = self.dropout(dialog_output)

        node_nums = edu_nums + 1
        edu_attn_mask = torch.arange(node_num).expand(len(node_nums), node_num).cuda() < node_nums.unsqueeze(1)
        edu_attn_mask = edu_attn_mask.unsqueeze(1).expand(batch_size, node_num, node_num).\
                        reshape(batch_size * node_num,node_num)
        edu_attn_mask = StructureAwareAttention.masking_bias(edu_attn_mask)

        nodes = dialog_output.unsqueeze(1).expand(batch_size, node_num, node_num, self.hidden_size) # B N N H
        nodes = nodes.reshape(batch_size * node_num, node_num, self.hidden_size)
        const_path = self.path_emb(speakers, turns).unsqueeze(1).expand(batch_size, node_num, node_num, node_num,
                     self.path_hidden_size).reshape(batch_size * node_num, node_num, node_num, self.path_hidden_size)
        struct_path = self.expand_and_mask_paths(self.path_model(graphs))
        memory = []
        # only update the one to predict
        update_mask = self.get_update_mask(batch_size, node_num)
        for _ in range(self.layer_num):
            nodes = self.gnn(nodes, edu_attn_mask, StudentModel.dns2ds(struct_path) + const_path)
            struct_path = self.path_update(nodes, const_path, struct_path, update_mask)
            layer_path = struct_path.reshape(batch_size, node_num, node_num, node_num, self.path_hidden_size)
            layer_path = self.get_hidden_state(layer_path)
            memory.append(layer_path)
            struct_path[update_mask] = self.dropout(struct_path[update_mask])
        tmp = torch.arange(node_num)
        struct_path = struct_path.reshape(batch_size, node_num, node_num, node_num, self.path_hidden_size)
        predicted_path = struct_path[:, tmp, tmp]
        multi_agent = torch.cat((dialog_input, dialog_output), dim=-1)
        expand_multi_agent = multi_agent.unsqueeze(1).expand(batch_size, node_num, node_num,
                                                             multi_agent.size(-1))
        multi_actors = torch.cat((expand_multi_agent, expand_multi_agent.transpose(1, 2), predicted_path), dim=-1)

        return self.link_classifier(multi_actors).squeeze(-1), self.label_classifier(multi_actors), memory

    def get_hidden_state(self, struct_path):
        batch_size, node_num, _, _, path_hidden_size = struct_path.size()
        hidden_state = torch.zeros(batch_size, node_num, node_num, path_hidden_size).to(struct_path)
        for i in range(1, node_num):
            hidden_state[:, i, :i] = struct_path[:, i, i, :i]
            hidden_state[:, :i, i] = struct_path[:, i, :i, i]
        return hidden_state

    def get_update_mask(self, batch_size, node_num):
        paths = torch.zeros(batch_size, node_num, node_num, node_num).bool()
        for i in range(node_num):
            paths[:, i, i, :i] = True
        return paths.reshape(batch_size * node_num, node_num, node_num)

    def expand_and_mask_paths(self, paths):
        batch_size, node_num, _, path_hidden_size = paths.size()
        paths = paths.unsqueeze(1).expand(batch_size, node_num, node_num, node_num, path_hidden_size).clone()
        for i in range(node_num):
            paths[:, i, i, :i] = 0
        return paths.reshape(batch_size * node_num, node_num, node_num, path_hidden_size)


class StructureAwareAttention(nn.Module):
    def __init__(self, hidden_size, path_hidden_size, head_num, dropout):
        super(StructureAwareAttention, self).__init__()
        self.q_transform = nn.Linear(hidden_size, hidden_size)
        self.k_transform = nn.Linear(hidden_size, hidden_size)
        self.v_transform = nn.Linear(hidden_size, hidden_size)
        self.struct_k_transform = nn.Linear(path_hidden_size, hidden_size // head_num)
        self.struct_v_transform = nn.Linear(path_hidden_size, hidden_size // head_num)
        self.o_transform = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.ReLU()
        self.hidden_size = hidden_size
        self.head_num = head_num
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_size)
        self.path_norm = nn.LayerNorm(path_hidden_size)

    def forward(self, nodes, bias, paths):
        q, k, v = self.q_transform(nodes), self.k_transform(nodes), self.v_transform(nodes)
        q = self.split_heads(q, self.head_num)
        k = self.split_heads(k, self.head_num)
        v = self.split_heads(v, self.head_num)
        paths = self.path_norm(paths)
        struct_k, struct_v = self.struct_k_transform(paths), self.struct_v_transform(paths)
        q = q * (self.hidden_size // self.head_num) ** -0.5
        w = torch.matmul(q, k.transpose(-1, -2)) + torch.matmul(q.transpose(1,2), struct_k.transpose(-1, -2)).transpose(1,2) + bias
        w = torch.nn.functional.softmax(w, dim=-1)
        output = torch.matmul(w, v)+torch.matmul(w.transpose(1,2), struct_v).transpose(1,2)
        output = self.activation(self.o_transform(self.combine_heads(output)))
        return self.norm(nodes + self.dropout(output))

    @staticmethod
    def split_heads(x, heads):
        batch = x.shape[0]
        length = x.shape[1]
        channels = x.shape[2]

        y = torch.reshape(x, [batch, length, heads, channels // heads])
        return torch.transpose(y, 2, 1)

    @staticmethod
    def combine_heads(x):
        batch = x.shape[0]
        heads = x.shape[1]
        length = x.shape[2]
        channels = x.shape[3]

        y = torch.transpose(x, 2, 1)

        return torch.reshape(y, [batch, length, heads * channels])

    @staticmethod
    def masking_bias(mask, inf=-1e9):
        ret = ~mask * inf
        return torch.unsqueeze(torch.unsqueeze(ret, 1),1)


# class NormalAttention(nn.Module):
#     def __init__(self, hidden_size, head_num, dropout):
#         super(NormalAttention, self).__init__()
#         self.q_transform = nn.Linear(hidden_size, hidden_size)
#         self.k_transform = nn.Linear(hidden_size, hidden_size)
#         self.v_transform = nn.Linear(hidden_size, hidden_size)
#         self.o_transform = nn.Linear(hidden_size, hidden_size)
#         self.activation = nn.ReLU()
#         self.hidden_size = hidden_size
#         self.head_num = head_num
#         self.dropout = nn.Dropout(dropout)
#         self.norm = nn.LayerNorm(hidden_size)
#
#     def forward(self, nodes, bias, paths):
#         q, k, v = self.q_transform(nodes), self.k_transform(nodes), self.v_transform(nodes)
#         q = self.split_heads(q, self.head_num)
#         k = self.split_heads(k, self.head_num)
#         v = self.split_heads(v, self.head_num)
#         paths = self.path_norm(paths)
#         q = q * (self.hidden_size // self.head_num) ** -0.5
#         w = torch.matmul(q, k.transpose(-1, -2))+bias
#         w = torch.nn.functional.softmax(w, dim=-1)
#         output = torch.matmul(w, v)
#         output = self.activation(self.o_transform(self.combine_heads(output)))
#         return self.norm(nodes + self.dropout(output))
#
#     @staticmethod
#     def split_heads(x, heads):
#         batch = x.shape[0]
#         length = x.shape[1]
#         channels = x.shape[2]
#
#         y = torch.reshape(x, [batch, length, heads, channels // heads])
#         return torch.transpose(y, 2, 1)
#
#     @staticmethod
#     def combine_heads(x):
#         batch = x.shape[0]
#         heads = x.shape[1]
#         length = x.shape[2]
#         channels = x.shape[3]
#
#         y = torch.transpose(x, 2, 1)
#
#         return torch.reshape(y, [batch, length, heads * channels])
#
#     @staticmethod
#     def masking_bias(mask, inf=-1e9):
#         ret = ~mask * inf
#         return torch.unsqueeze(torch.unsqueeze(ret, 1),1)

class PathUpdateModel(nn.Module):
    def __init__(self, params):
        super(PathUpdateModel, self).__init__()
        self.x_dim=params.hidden_size
        self.h_dim=params.path_hidden_size

        self.r = nn.Linear(2*self.x_dim + self.h_dim, self.h_dim, True)
        self.z = nn.Linear(2*self.x_dim + self.h_dim, self.h_dim, True)

        self.c = nn.Linear(2*self.x_dim, self.h_dim, True)
        self.u = nn.Linear(self.h_dim, self.h_dim, True)

    def forward(self, nodes, bias, hx, mask=None):
        batch_size, node_num, hidden_size = nodes.size()
        nodes = nodes.unsqueeze(1).expand(batch_size, node_num, node_num, hidden_size)
        nodes = torch.cat((nodes, nodes.transpose(1, 2)),dim=-1) # B N N H
        if hx is None:
            hx=torch.zeros_like(bias)
        if mask is not None:
            nodes, bias =nodes[mask], bias[mask]

        rz_input = torch.cat((nodes, hx[mask]), -1)
        r = torch.sigmoid(self.r(rz_input))
        z = torch.sigmoid(self.z(rz_input))

        u = torch.tanh(self.c(nodes) + r * self.u(hx[mask]))

        hx[mask] = z * hx[mask] + (1 - z) * u
        return hx

# class PathUpdateModel(nn.Module):
#     def __init__(self, params):
#         super(PathUpdateModel, self).__init__()
#         self.gru = nn.GRU(params.hidden_size + params.path_hidden_size, params.path_hidden_size, batch_first=True)
#
#     def forward(self, nodes, bias, hx=None):
#         batch_size, node_num, hidden_size = nodes.size()
#         path_hidden_size = bias.size(-1)
#         nodes = nodes.unsqueeze(1).expand(batch_size, node_num, node_num, hidden_size)
#         nodes = nodes - nodes.transpose(1, 2)
#         nodes = torch.cat((nodes, bias), dim=-1)
#         output, hx = self.gru(nodes.reshape(-1, 1, hidden_size + path_hidden_size), hx)
#         return output.reshape(batch_size, node_num, node_num, path_hidden_size), hx


class PathEmbedding(nn.Module):
    def __init__(self, params):
        super(PathEmbedding, self).__init__()
        self.speaker = nn.Embedding(2, params.path_hidden_size // 4)
        self.turn = nn.Embedding(2, params.path_hidden_size // 4)
        self.valid_dist = params.valid_dist
        self.position = nn.Embedding(self.valid_dist * 2 + 3, params.path_hidden_size // 2)

        self.tmp = torch.arange(200)
        self.path_pool = self.tmp.expand(200, 200) - self.tmp.unsqueeze(1)
        self.path_pool[self.path_pool > self.valid_dist] = self.valid_dist + 1
        self.path_pool[self.path_pool < -self.valid_dist] = -self.valid_dist - 1
        self.path_pool += self.valid_dist + 1

    def forward(self, speaker, turn):
        batch_size, node_num, _ = speaker.size()
        # a, b = speaker == 1, turn == 1
        # c = torch.zeros_like(speaker)
        # c[a & ~b] = 1
        # c[~a & b] = 2
        # c[a & b] = 3
        speaker = self.speaker(speaker)
        turn = self.turn(turn)
        position = self.position(self.path_pool[:node_num, :node_num].cuda())
        position = position.expand(batch_size, node_num, node_num, position.size(-1))
        return torch.cat((speaker, turn, position), dim=-1)


class PathClassifier(nn.Module):
    def __init__(self, params):
        super(PathClassifier, self).__init__()
        self.type_num = params.relation_type_num
        self.classifier = Classifier(params.path_hidden_size, params.path_hidden_size, params.relation_type_num)

    def forward(self, path, target, mask):
        path = self.classifier(path)[mask]
        target = target[mask]
        weight = torch.ones(self.type_num).float().cuda()
        weight[0] /= target.size(0)**0.5
        return torch.nn.functional.cross_entropy(path, target, weight=weight, reduction='mean')


class PathModel(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.path_hidden_size=params.path_hidden_size
        self.type_num=params.relation_type_num
        self.spec_type = nn.Parameter(torch.zeros(1, params.path_hidden_size), requires_grad=False)
        self.normal_type = nn.Parameter(torch.empty(params.relation_type_num - 1, params.path_hidden_size),
                                        requires_grad=True)
        self.dropout=nn.Dropout(0.1)
        self.reset_parameters()

    def forward(self, graphs):
        label_embedding = torch.cat((self.spec_type, self.normal_type), dim=0)
        # graphs=self.mask_graphs(graphs)
        graphs=graphs+graphs.transpose(1,2)
        path = self.dropout(nn.functional.embedding(graphs, weight=label_embedding, padding_idx=0))
        return path

    def mask_graphs(self, graphs):
        graphs = graphs.clone()
        batch_size, node_num, _ = graphs.size()
        mask = torch.rand(batch_size, node_num) > 0.5
        graphs[~mask] = 0
        return graphs

    def reset_parameters(self):
        nn.init.normal_(self.normal_type, mean=0.0,
                        std=self.path_hidden_size ** -0.5)


class Classifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_class):
        super().__init__()
        self.input_transform = nn.Sequential(nn.Linear(input_size, hidden_size), nn.Tanh())
        self.output_transform = nn.Linear(hidden_size, num_class)

    def forward(self, x):
        return self.output_transform(self.input_transform(x))


class Bridge(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.bridge=nn.Linear(params.path_hidden_size, params.path_hidden_size)

    def forward(self, input):
        return self.bridge(input)