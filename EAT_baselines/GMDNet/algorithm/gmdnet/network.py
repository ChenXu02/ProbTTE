import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
from gnn_layers import ResidualGatedGCNLayer

class SelfAttention(nn.Module):
    def __init__(self, att_hidden_size, num_of_attention_heads):
        super().__init__()

        self.num_attention_heads = num_of_attention_heads
        self.attention_head_size = int(att_hidden_size / num_of_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(att_hidden_size , self.all_head_size)
        self.key = nn.Linear(att_hidden_size , self.all_head_size)
        self.value = nn.Linear(att_hidden_size, self.all_head_size)
        self.dense = nn.Linear(att_hidden_size , att_hidden_size)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, r, mask):
        mixed_query_layer = self.query(r)
        mixed_key_layer = self.key(r)
        mixed_value_layer = self.value(r)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        mask = torch.repeat_interleave(mask.unsqueeze(1), repeats=self.num_attention_heads, dim=1)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # mask
        attention_scores = attention_scores.masked_fill(mask == 0, 1e-8)
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        output = self.dense(context_layer)

        return output

class MultiLayerPerceptron(torch.nn.Module):

    def __init__(self, input_dim, embed_dims, dropout=0.2, output_layer=True):
        super(MultiLayerPerceptron, self).__init__()
        layers = list()
        for embed_dim in embed_dims:
            layers.append(torch.nn.Linear(input_dim, embed_dim))
            layers.append(torch.nn.BatchNorm1d(embed_dim))
            layers.append(torch.nn.ReLU())
            if dropout > 0:
                layers.append(torch.nn.Dropout(p=dropout))
            input_dim = embed_dim
        if output_layer:
            layers.append(torch.nn.Linear(input_dim, 1))
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)

class Encoder(nn.Module):
    def __init__(self,
                 hidden_dim,
                 node_dim,
                 voc_edges_in,
                 aggregation,
                 edges_values_dim,
                 time_embed_dim,
                 edge_out_dim,
                 route_fea_dim,
                 gnn_num_layers,
                 att_hidden_size,
                 num_of_attention_heads,
                 max_seq_len
                 ):
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.node_dim = node_dim
        self.voc_edges_in = voc_edges_in
        self.aggregation = aggregation
        self.edge_values_dim = edges_values_dim
        self.time_embed_dim = time_embed_dim
        self.edge_out_dim = edge_out_dim
        self.route_fea_dim = route_fea_dim
        self.gnn_num_layers = gnn_num_layers
        self.att_hidden_size = att_hidden_size
        self.max_seq_len = max_seq_len
        gnn_layers = []
        for layer in range(self.gnn_num_layers):
           gnn_layers.append(ResidualGatedGCNLayer(self.hidden_dim, self.aggregation))
        self.mlp = MultiLayerPerceptron(self.route_fea_dim + self.att_hidden_size * self.max_seq_len, (self.hidden_dim,), dropout=0.2, output_layer=False)
        self.nodes_embedding = nn.Linear(self.node_dim, self.hidden_dim, bias=False)
        self.edges_embedding = nn.Embedding(self.voc_edges_in, self.hidden_dim // 2)
        self.edges_values_embedding = nn.Linear(self.edge_values_dim, self.hidden_dim // 2, bias=False)
        self.route_update = SelfAttention(self.att_hidden_size, num_of_attention_heads)
        self.edge_linear = nn.Linear(self.hidden_dim, self.edge_out_dim)
        self.gcn_layers = nn.ModuleList(gnn_layers)
        self.time_embed = nn.Embedding(24, self.time_embed_dim)

    def positional_encoding(self):
        pe_table = []
        for pos in range(self.max_seq_len):
            pos_en = []
            for ii in range(0, self.edge_out_dim, 2):
                pos_en.append(math.sin(pos / 10000 ** (2 * ii / self.edge_out_dim )))
                pos_en.append(math.cos(pos / 10000 ** (2 * ii / self.edge_out_dim )))
            pe_table.append(pos_en)
        return torch.FloatTensor(pe_table)

    def forward(self, f, route, mask, edge, node, A):
        B = f.shape[0]
        N = edge.shape[1]
        #Select the graph based on the time slice
        time_index = f[:, 0]
        edge = edge[time_index.long()]
        node = node[time_index.long()]
        A = A[time_index.long()]

        #initial node_embedding
        node_embed = self.nodes_embedding(node[:, :, 1:])
        #initial edge_embedding
        e_vals = self.edges_values_embedding(edge[:, :, :, 2:])
        e_tags = self.edges_embedding(A)
        e = torch.cat((e_vals, e_tags), dim=3)
        #Spatial Dependency Modeling
        for layer in range(self.gnn_num_layers):
            nodes_embed, e = self.gcn_layers[layer](node_embed, e)
        edge = self.edge_linear(e).reshape(B, N, N, -1)  # [B, N, N, H]

        # Look up edge embedding to obtain route embedding
        from_node_index = route[:, :, 0]
        to_node_index = route[:, :, 1]
        R = edge[:, from_node_index, to_node_index, :].diagonal(dim1=0, dim2=1).permute(2, 0, 1).contiguous()

        # Position Embedding
        PE = torch.repeat_interleave(self.positional_encoding().unsqueeze(0), repeats=B, dim=0).to(f.device)  # [B, seq_len, dim]
        R = torch.cat([R, PE], dim=2)
        t_s = self.time_embed(f[:, 1].unsqueeze(1).unsqueeze(1).repeat(1, route.shape[1], 1).long()).squeeze(2)
        R = torch.cat([R, t_s], dim=2).reshape(B, self.max_seq_len, self.edge_out_dim * 2 +  t_s.shape[-1])

        # Mutual Correlation Modeling
        r = self.route_update(R, mask)
        r = torch.cat([r.reshape(r.shape[0], -1), f[:, 1:]], dim=1)
        return r

class Decoder(nn.Module):
    def __init__(self, hidden_dim, input_dim, n_gaussians):
        super(Decoder, self).__init__()
        self.phi_h = MultiLayerPerceptron(input_dim, (hidden_dim,), dropout=0.2, output_layer=False)
        self.phi_mu = nn.Linear(hidden_dim, n_gaussians)
        self.phi_pi = nn.Linear(hidden_dim, n_gaussians)
        self.phi_sigma = nn.Linear(hidden_dim, n_gaussians)

    def forward(self, r):
        r_h = self.phi_h(r)
        pi = F.softmax(self.phi_pi(r_h), -1)
        mu = self.phi_mu(r_h)
        sigma = torch.exp(self.phi_sigma(r_h))
        return pi, mu, sigma


class GMDNet(nn.Module):

    def __init__(self, args):
        super(GMDNet, self).__init__()
        self.config = args
        self.hidden_dim = args.get('hidden_dim', 16)
        self.node_dim = args.get('node_dim', 3)
        self.voc_edges_in = args.get('voc_edges_in', 3)
        self.aggregation = args.get('aggregation', "mean")
        self.edge_values_dim = args.get('edge_dim', 5)
        self.time_embed_dim = args.get('time_embed_dim', 8)
        self.edge_out_dim = int((args['att_hidden_size'] - self.time_embed_dim) / 2)
        self.route_fea_dim = args.get('route_fea_dim', 8)
        self.gnn_num_layers = args['num_layers']
        self.n_gaussians = args['n_gaussians']
        self.max_seq_len = args.get('max_seq_len', 4)
        self.att_hidden_size = args['att_hidden_size']
        self.num_of_attention_heads = args['num_of_attention_heads']
        self.dirichlet_alpha = args['dirichlet_alpha']

        self.route_encoder = Encoder(
            self.hidden_dim,
            self.node_dim,
            self.voc_edges_in,
            self.aggregation,
            self.edge_values_dim,
            self.time_embed_dim,
            self.edge_out_dim,
            self.route_fea_dim,
            self.gnn_num_layers,
            self.att_hidden_size,
            self.num_of_attention_heads,
            self.max_seq_len,
        )

        self.decoder = Decoder(
            self.hidden_dim,
            self.route_fea_dim + self.att_hidden_size * self.max_seq_len,
            self.n_gaussians
        )

    def E_step(self, pi, mu, sigma, labels, dirichlet_alpha):
        # implement the E_step using ground truth
        num_components = pi.shape[1]
        dirichlet = torch.distributions.dirichlet.Dirichlet(torch.tensor([dirichlet_alpha] * num_components, dtype=torch.float32).to(labels.device))
        m = torch.distributions.Normal(loc=mu, scale=sigma)
        emission_of_true_labels = torch.exp(m.log_prob(labels.float()))
        unnorm_posterior_estimate = torch.mul(emission_of_true_labels, pi) + (1e-8)
        posterior_estimate = unnorm_posterior_estimate / unnorm_posterior_estimate.sum(dim=1, keepdim=True)
        E_log_likelihood = torch.mean(torch.mul(posterior_estimate, unnorm_posterior_estimate.log()))
        prior_term = torch.mean(dirichlet.log_prob(pi), dim=0)
        objective = E_log_likelihood + prior_term
        return -objective

    def M_step(self):
        pass # Implement the M_step by optimizer

    def forward(self, route, mask, f, edge, node, A, labels, mode):
        '''
         :param route: (batch, seq_len, d_od)
         :param mask: (batch, seq_len, seq_len)
         :param f: (batch, d_f)
         :param edge: (T, N, N, d_edge)
         :param node:  (T, N, d_node)
         :param labels: (batch, d_label)
         :param mode: train / test
         :return: objective function / mixture weights and mixture components
         '''
        if mode == 'train':
            # Graph-cooperated Route Encoding Layer
            r = self.route_encoder(f, route, mask, edge, node, A)
            # Mixture Density Decoding Layer
            pi, mu, sigma = self.decoder(r)
            return self.E_step(pi, mu, sigma, labels, self.dirichlet_alpha)
        else:
            # Graph-cooperated Route Encoding Layer
            r = self.route_encoder(f, route, mask, edge, node, A)
            # Mixture Density Decoding Layer
            pi, mu, sigma = self.decoder(r)
            return pi, mu, sigma

    def model_file_name(self):
        t = time.time()
        file_name = '+'.join([f'{k}-{self.config[k]}' for k in ['hidden_dim']])
        file_name = f'{file_name}.{t}'
        return file_name
