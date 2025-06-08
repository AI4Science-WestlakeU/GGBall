"""
The code is taken from https://github.com/lucidrains/vector-quantize-pytorch
"""
import copy
import torch
from torch import nn, einsum
import torch.nn.functional as F
import torch.distributed as distributed
from torch.cuda.amp import autocast
from torch.distributions.categorical import Categorical
from typing import Union
from einops import rearrange, repeat
from contextlib import contextmanager
from src.models.hyperbolic_nn_plusplus.geoopt_plusplus.manifolds import PoincareBall
from src.distribution.wrapped_normal import WrappedNormalPoincare
from geoopt import ManifoldParameter
import pdb


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def noop(*args, **kwargs):
    pass


def l2norm(t):
    return F.normalize(t, p=2, dim=-1)


def log(t, eps=1e-20):
    return torch.log(t.clamp(min=eps))


def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))


def gumbel_sample(t, temperature=1., dim=-1):
    if temperature == 0:
        return t.argmax(dim=dim)

    return ((t / temperature) + gumbel_noise(t)).argmax(dim=dim)


def softmax_sample(t, temperature, dim=-1):
    if isinstance(temperature, type(None)) or (temperature == 'None'):
        return t.argmax(dim=dim)

    m = Categorical(logits=t / temperature)
    return m.sample()


def ema_inplace(moving_avg, new, decay):
    moving_avg.data.mul_(decay).add_(new, alpha=(1 - decay))


def laplace_smoothing(x, n_categories, eps=1e-5):
    return (x + eps) / (x.sum() + n_categories * eps)


def sample_vectors(samples, num):
    num_samples, device = samples.shape[0], samples.device

    if num_samples >= num:
        indices = torch.randperm(num_samples, device=device)[:num]
    else:
        indices = torch.randint(0, num_samples, (num,), device=device)

    return samples[indices]


def kmeans(samples, num_clusters, num_iters=10, use_cosine_sim=False):
    dim, dtype, device = samples.shape[-1], samples.dtype, samples.device

    means = sample_vectors(samples, num_clusters)

    for _ in range(num_iters):
        if use_cosine_sim:
            dists = samples @ means.t()
        else:
            diffs = rearrange(samples, 'n d -> n () d') - rearrange(means, 'c d -> () c d')
            dists = -(diffs ** 2).sum(dim=-1)

        buckets = dists.max(dim=-1).indices
        bins = torch.bincount(buckets, minlength=num_clusters)
        zero_mask = bins == 0
        bins_min_clamped = bins.masked_fill(zero_mask, 1)

        new_means = buckets.new_zeros(num_clusters, dim, dtype=dtype)
        new_means.scatter_add_(0, repeat(buckets, 'n -> n d', d=dim), samples)
        new_means = new_means / bins_min_clamped[..., None]

        if use_cosine_sim:
            new_means = l2norm(new_means)

        means = torch.where(zero_mask[..., None], means, new_means)

    return means, bins


@torch.no_grad()
def hyperbolic_mean(p, xs, w, manifold):
    # p: (K,D) 上一步码字   xs: (M,D) 属于该码字的样本  w: 权重或1
    v = manifold.logmap(xs, p[:,None,:])        # -> (K,M,D) 切向量
    v = (v * w[None,:,None]).mean(1)            # 加权切向量平均
    return manifold.expmap(v, p)                # 测地前进到新均值

@torch.no_grad()
def hyperbolic_ema(p, q, decay, manifold):
    v = manifold.logmap(q, p)                   # ① 取 Log_p(q)
    p_new = manifold.expmap((1-decay)*v, p)     # ② 沿测地走 (1-α) 步
    return manifold.projx(p_new)                # ③ 投回球内


# regularization losses

def orthgonal_loss_fn(t):
    # eq (2) from https://arxiv.org/abs/2112.00384
    n = t.shape[0]
    normed_codes = l2norm(t)
    identity = torch.eye(n, device=t.device)
    cosine_sim = einsum('i d, j d -> i j', normed_codes, normed_codes)
    return ((cosine_sim - identity) ** 2).sum() / (n ** 2)


# distance types

class Codebook(nn.Module):
    """
    source: https://github.com/lucidrains/vector-quantize-pytorch/blob/master/vector_quantize_pytorch/vector_quantize_pytorch.py
    """
    def __init__(
            self,
            dim,
            codebook_size,
            kmeans_init=False,
            kmeans_iters=10,
            decay=0.8,
            eps=1e-5,
            threshold_ema_dead_code=2,
            use_ddp=False,
            learnable_codebook=True,
            sample_codebook_temp=0,
            emb_dropout=0.,
            use_hyperbolic = False,
            manifold=None,
    ):
        super().__init__()
        
        self.use_hyperbolic = use_hyperbolic
        if self.use_hyperbolic:
            self.manifold = manifold
            
        self.decay = decay
        self.learnable_codebook = learnable_codebook
        self.codebook_size = codebook_size
        self.dim = dim
        self._build_codebook()
        # init_fn = torch.randn if not kmeans_init else torch.zeros
        # # 采codebooksize个
        # embed = init_fn(codebook_size, dim)
        
        # # FIXME: Normal -> Uniform
        # # embed = WrappedNormalPoincare(
        # #     torch.zeros((codebook_size, dim)),
        # #     torch.ones((codebook_size, dim)),
        # #     self.manifold
        # # ).rsample()
        # embed = torch.randn((codebook_size, dim))
        # import pdb; pdb.set_trace()
        # 打印初始embed的norm
        embed_norm = torch.norm(self.embed, p=2, dim=1)
        print(f"Initial embed norm shape: {embed_norm.shape}, embed norm min: {embed_norm.min().item():.4f}, embed norm max: {embed_norm.max().item():.4f}, embed norm mean: {embed_norm.mean().item():.4f}")

        self.kmeans_iters = kmeans_iters
        self.eps = eps
        self.threshold_ema_dead_code = threshold_ema_dead_code
        self.sample_codebook_temp = sample_codebook_temp
        self.emb_dropout = emb_dropout

        self.all_reduce_fn = distributed.all_reduce if use_ddp else noop

        self.register_buffer('initted', torch.Tensor([not kmeans_init]))
        self.register_buffer('cluster_size', torch.zeros(codebook_size))
        self.kmeans_init = kmeans_init
    
        self.embed_onehot = None
        self.perplexity = None


    def manifold_kmeans(self, samples, num_clusters, num_iters=10):
        # 使用 geodesic 距离；手动负平方方便与欧氏原实现一致
        dist2 = lambda a, b: -self.manifold.dist(a, b)**2      # (N,C)

        means = sample_vectors(samples, num_clusters)          # 随机初值
        for _ in range(num_iters):
            dists = dist2(samples.unsqueeze(1), means.unsqueeze(0))
            buckets = dists.max(dim=-1).indices
            bins = torch.bincount(buckets, minlength=num_clusters)

            zero_mask = bins == 0
            bins_safe = bins.masked_fill(zero_mask, 1)

            # new_means = buckets.new_zeros(num_clusters, samples.size(-1), dtype=samples.dtype)
            # new_means.scatter_add_(0,
            #                     repeat(buckets, 'n -> n d', d=samples.size(-1)),
            #                     samples)
            # new_means = self.manifold.projx(new_means / bins_safe.unsqueeze(-1))
            # 创建one-hot编码来表示每个样本属于哪个桶
            embed_onehot = F.one_hot(buckets, num_clusters).float()  # (N, K)
            
            # 使用weighted_midpoint计算新的聚类中心
            # 准备输入数据: (K, N, D) 和权重: (K, N)
            xs = samples.unsqueeze(0).expand(num_clusters, -1, -1)  # (K, N, D)
            weights = embed_onehot.t()  # (K, N)
            
            # 计算加权中点
            new_means = self.manifold.weighted_midpoint(
                xs=xs,
                weights=weights,
                reducedim=[1],  # 在样本维度上进行平均
                keepdim=False,  # 去掉被平均的轴
                lincomb=False,
            )

            # 处理空桶
            means = torch.where(zero_mask[..., None], means, new_means)

        return means, bins

    # @torch.jit.ignore
    def init_embed_(self, data):
        if self.initted:
            return
        if self.kmeans_init:
            embed, cluster_size = self.manifold_kmeans(data, self.codebook_size, self.kmeans_iters)
        else:
            embed = self.embed.data
            ## calculate cluster_size beween data and embed use dist_matmul
            dist = self.manifold.dist_matmul(
                data.unsqueeze(0),
                embed.unsqueeze(0)
            )
            cluster_size = dist.sum(dim=-1)
            cluster_size = cluster_size / cluster_size.sum()

        self.embed.data.copy_(embed)
        self.embed_avg.data.copy_(embed.clone())
        self.cluster_size.data.copy_(cluster_size)
        self.initted.data.copy_(torch.Tensor([True]))

    def _build_codebook(self):
        # embed = torch.randn(self.codebook_size, self.dim)
        # embed = embed / embed.norm(dim=-1, keepdim=True)

        # # Sample norms uniformly between 0 and 1
        # sampled_norms = torch.rand(self.codebook_size, device=embed.device)
        # embed = embed * sampled_norms.unsqueeze(-1)

        embed = self.manifold.random(self.codebook_size, self.dim)

        # if self.use_hyperbolic:
        #     self.embed = ManifoldParameter(embed, manifold=self.manifold)
        # else:
        #     self.embed = nn.Parameter(embed)

        if self.learnable_codebook:
            self.embed = ManifoldParameter(embed, manifold=self.manifold)
        else:
            self.register_buffer('embed', embed)
        self.register_buffer('embed_avg', embed.clone())

    def replace(self, samples, mask):
        modified_codebook = torch.where(
            mask[..., None],
            sample_vectors(samples, self.codebook_size),
            self.embed
        )
        self.embed.data.copy_(modified_codebook)

    def expire_codes_(self, batch_samples):
        if self.threshold_ema_dead_code == 0:
            return

        expired_codes = self.cluster_size < self.threshold_ema_dead_code
        if not torch.any(expired_codes):
            return
        ## 查看有多少个expired_codes
        print(f"expired_codes: {expired_codes.sum().item()}")
        batch_samples = rearrange(batch_samples, '... d -> (...) d')
        self.replace(batch_samples, mask=expired_codes)

    @autocast(enabled=False)
    def forward(self, x, svq_temp:Union[float,None]=None, node_mask=None):
        # import pdb; pdb.set_trace()
        shape, dtype = x.shape, x.dtype
        flatten = rearrange(x, '... d -> (...) d')

        self.init_embed_(flatten)

        # embed = self.embed if not self.learnable_codebook else self.embed.detach()
        embed = self.embed
        embed = embed.t()

        if self.emb_dropout and self.training:
            embed = F.dropout(embed, self.emb_dropout)

        # 打印flatten的norm
        flatten_norm = torch.norm(flatten, p=2, dim=1)
        print(f"flatten norm shape: {flatten_norm.shape}, flatten norm min: {flatten_norm.min().item():.4f}, flatten norm max: {flatten_norm.max().item():.4f}, flatten norm mean: {flatten_norm.mean().item():.4f}")
        if self.use_hyperbolic:
            dist = -self.manifold.dist_matmul(
                flatten.unsqueeze(0),
                embed.unsqueeze(0)
            )
            # 打印dist和embed的norm
            embed_norm = torch.norm(embed, p=2, dim=0)
            print(f"dist shape: {dist.shape}, dist min: {dist.min().item():.4f}, dist max: {dist.max().item():.4f}, dist mean: {dist.mean().item():.4f}")
            print(f"embed norm shape: {embed_norm.shape}, embed norm min: {embed_norm.min().item():.4f}, embed norm max: {embed_norm.max().item():.4f}, embed norm mean: {embed_norm.mean().item():.4f}")
        else:
            dist = -(
                flatten.pow(2).sum(1, keepdim=True)
                - 2 * flatten @ embed
                + embed.pow(2).sum(0, keepdim=True)
            )
            # 打印dist和embed的norm
            embed_norm = torch.norm(embed, p=2, dim=0)
            print(f"dist shape: {dist.shape}, dist min: {dist.min().item():.4f}, dist max: {dist.max().item():.4f}, dist mean: {dist.mean().item():.4f}")
            print(f"embed norm shape: {embed_norm.shape}, embed norm min: {embed_norm.min().item():.4f}, embed norm max: {embed_norm.max().item():.4f}, embed norm mean: {embed_norm.mean().item():.4f}")

        # embed_ind = gumbel_sample(dist, dim=-1, temperature=self.sample_codebook_temp)
        temp = svq_temp
        # import pdb; pdb.set_trace()
        if self.training:
            embed_ind = softmax_sample(dist, dim=-1, temperature=self.sample_codebook_temp)
            # TODO: 现在是softmax而不是argmax
        else:
            embed_ind = softmax_sample(dist, dim=-1, temperature=None)  # no stochasticity
        embed_onehot = F.one_hot(embed_ind, self.codebook_size).type(dtype).squeeze(0)
        node_mask = rearrange(node_mask, '... -> (...)').unsqueeze(-1)
        embed_onehot = embed_onehot * node_mask

        embed_ind = embed_ind.view(*shape[:-1])
        quantize = F.embedding(embed_ind, self.embed)
        if not self.learnable_codebook and self.training:
            # import pdb; pdb.set_trace()
            cluster_size = embed_onehot.sum(0)                       # (K,)
            ema_inplace(self.cluster_size, cluster_size, self.decay) # 原逻辑保留

            # --------- 利用 weighted_midpoint_bmm 求 K 个 Fréchet-mean -------- #
            # xs : (K, N, D)   weights : (K, N)
            xs = flatten.unsqueeze(0).expand(self.codebook_size, -1, -1)
            weights = embed_onehot.t()                               # (K, N)

            new_centroids = self.manifold.weighted_midpoint(
                    xs = xs,
                    weights = weights,
                    reducedim = [1],   # 告诉函数「第 1 维 (N) 要被平均」
                    keepdim = False, # 去掉被平均的轴
                    lincomb = False,
            )
            # ---------------------------------------------------------------
            # ② 用 Einstein midpoint 进行 β-EMA  (两点、权重 β : 1-β)
            beta = float(self.decay)
            two_pts = torch.stack((self.embed_avg, new_centroids), dim=1)  # (K, 2, D)
            w = torch.tensor([beta, 1-beta], device=two_pts.device)  # (2,)

            smooth_centroids = self.manifold.weighted_midpoint(
                    xs = two_pts,
                    weights = w.expand(self.codebook_size, 2),   # (K, 2)
                    reducedim = [1],                               # 把「2」那一维平均掉
                    keepdim = False,
            )

            # 投影回球内，完成更新
            self.embed.data.copy_( self.manifold.projx(smooth_centroids) )
            self.expire_codes_(x)                 # 死码字逻辑照旧
            # ---------------------------------------------------------------

            # cluster_size = embed_onehot.sum(0)
            # self.all_reduce_fn(cluster_size)
            # ema_inplace(self.cluster_size, cluster_size, self.decay)

            # embed_sum = flatten.t() @ embed_onehot
            # self.all_reduce_fn(embed_sum)

            # ema_inplace(self.embed_avg, embed_sum.t(), self.decay)
            # cluster_size = laplace_smoothing(self.cluster_size, self.codebook_size, self.eps) * self.cluster_size.sum()
            # embed_normalized = self.embed_avg / cluster_size.unsqueeze(1)
            # embed_normalized = self.manifold.projx(embed_normalized)
            # self.embed.data.copy_(embed_normalized)
            # self.expire_codes_(x)

        # 打印更新后的embed的norm
        # if self.training and self.learnable_codebook:
        embed_norm = torch.norm(self.embed, p=2, dim=1)
        print(f"Updated embed norm shape: {embed_norm.shape}, embed norm min: {embed_norm.min().item():.4f}, embed norm max: {embed_norm.max().item():.4f}, embed norm mean: {embed_norm.mean().item():.4f}")

        # perplexity
        avg_probs = torch.mean(embed_onehot, dim=0)  # (K,)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        self.embed_onehot = embed_onehot.detach()  # .cpu()
        self.perplexity = perplexity.detach()  # .cpu()

        # print(f"flatten: {flatten.norm(dim=-1)}")
        # print(f"embed: {embed.norm(dim=-1)}")
        # print(f"dist: {dist}")
        # print(f"quantize: {quantize.norm(dim=-1)}")
        # print(f"perplexity: {perplexity}")

        return quantize, embed_ind


# main class
class VectorQuantize(nn.Module):
    def __init__(
            self,
            dim,
            codebook_size,
            codebook_dim=None,
            heads=1,
            decay=0.8,
            eps=1e-5,
            kmeans_init=False,
            kmeans_iters=10,
            use_cosine_sim=False,
            threshold_ema_dead_code=5,
            channel_last=True,
            accept_image_fmap=False,
            vq_loss_weight=1.,
            commitment_weight=1.,
            orthogonal_reg_weight=0.1,
            orthogonal_reg_active_codes_only=False,
            orthogonal_reg_max_codes=None,
            sample_codebook_temp=0,
            sync_codebook=False,
            emb_dropout=0.,
            use_hyperbolic=False,  # 新增参数
            manifold=None,
            **kwargs
    ):
        super().__init__()
        self.heads = heads
        codebook_dim = default(codebook_dim, dim)
        codebook_input_dim = codebook_dim * heads

        requires_projection = codebook_input_dim != dim
        self.project_in = nn.Linear(dim, codebook_input_dim) if requires_projection else nn.Identity()
        self.project_out = nn.Linear(codebook_input_dim, dim) if requires_projection else nn.Identity()

        self.eps = eps
        self.commitment_weight = commitment_weight
        self.vq_loss_weight = vq_loss_weight
        has_vq_loss_weight = vq_loss_weight > 0
        self.orthogonal_reg_weight = orthogonal_reg_weight
        self.orthogonal_reg_active_codes_only = orthogonal_reg_active_codes_only
        self.orthogonal_reg_max_codes = orthogonal_reg_max_codes
        self.learnable_codebook = has_vq_loss_weight
        # codebook_class = EuclideanCodebook

        # self._codebook = codebook_class(
        #     dim=codebook_dim,
        #     codebook_size=codebook_size,
        #     kmeans_init=kmeans_init,
        #     kmeans_iters=kmeans_iters,
        #     decay=decay,
        #     eps=eps,
        #     threshold_ema_dead_code=threshold_ema_dead_code,
        #     use_ddp=sync_codebook,
        #     learnable_codebook=has_codebook_orthogonal_loss,
        #     sample_codebook_temp=sample_codebook_temp,
        #     emb_dropout=emb_dropout,
        # )

        codebook_class = Codebook
        codebook_kwargs = dict(
            dim=codebook_dim,
            codebook_size=codebook_size,
            kmeans_init=kmeans_init,
            kmeans_iters=kmeans_iters,
            decay=decay,
            eps=eps,
            threshold_ema_dead_code=threshold_ema_dead_code,
            use_ddp=sync_codebook,
            learnable_codebook=has_vq_loss_weight,
            sample_codebook_temp=sample_codebook_temp,
            emb_dropout=emb_dropout,
        )
        if use_hyperbolic:
            codebook_kwargs['manifold'] = manifold
            codebook_kwargs['use_hyperbolic'] = use_hyperbolic
            self.manifold = manifold
        self._codebook = codebook_class(**codebook_kwargs)
        
        self.codebook_size = codebook_size

        self.accept_image_fmap = accept_image_fmap
        self.channel_last = channel_last

    @property
    def codebook(self):
        return self._codebook.embed

    def forward_hyp1(self, x, svq_temp=None, node_mask=None):
        # pdb.set_trace()
        shape = x.shape
        flatten = rearrange(x, '... d -> (...) d')

        embed = self._codebook.embed.t()
        dist = -self.manifold.dist_matmul(
            flatten.unsqueeze(0),
            embed.unsqueeze(0)
        )
        embed_ind = softmax_sample(dist, dim=-1, temperature=svq_temp)

        perplexity = None
        if node_mask is not None:
            embed_onehot = F.one_hot(embed_ind, self.codebook_size).type(x.dtype).squeeze(0)
            node_mask = rearrange(node_mask, '... -> (...)').unsqueeze(-1)
            embed_onehot = embed_onehot * node_mask
            avg_probs = torch.mean(embed_onehot, dim=0)  # (K,)
            perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        embed_ind = embed_ind.view(*shape[:-1])
        quantize = F.embedding(embed_ind, self._codebook.embed)

        return embed_ind, quantize, perplexity.item()
    
    def forward_hyp2(self, x, svq_temp=None, node_mask=None):
        # pdb.set_trace()
        shape = x.shape
        flatten = rearrange(x, '... d -> (...) d')

        embed = self._codebook.embed.t()
        dist = -self.manifold.dist_matmul(
            flatten.unsqueeze(0),
            embed.unsqueeze(0)
        )
        dist = dist.squeeze(0).view(*shape[:-1], -1)
        dist = dist/svq_temp
        
        return dist

    def random_sample(self, batch_size, prior_count=None):
        # sample index from 0 to codebook_size-1
        if prior_count is None:
            indices = torch.randint(0, self.codebook_size, (batch_size,)).to(self.codebook.device)
        elif prior_count.sum() == 0:
            indices = torch.randint(0, self.codebook_size, (batch_size,)).to(self.codebook.device)
        else:
            # sample from prior_count
            prior_count = prior_count / prior_count.sum()
            indices = torch.multinomial(prior_count, batch_size, replacement=True)
        # find corresponding codebook vector
        codebook_vectors = F.embedding(indices.to(self.codebook.device), self._codebook.embed)
        return codebook_vectors

    def forward(self, x, node_mask=None, svq_temp:Union[float,None]=None):
        """
        x: (B, N, D)
        """
        shape, device, heads, is_multiheaded, codebook_size = x.shape, x.device, self.heads, self.heads > 1, self.codebook_size
        need_transpose = not self.channel_last and not self.accept_image_fmap
        vq_loss = {'loss': torch.tensor([0.], device=device, requires_grad=self.training),
                   'vq_loss': 0.,
                   'commit_loss': 0.,
                   'orthogonal_reg_loss': 0.,
                   }
        edge_mask = node_mask.unsqueeze(-1) * node_mask.unsqueeze(-2)
        if self.accept_image_fmap:
            height, width = x.shape[-2:]
            x = rearrange(x, 'b c h w -> b (h w) c')

        if need_transpose:
            x = rearrange(x, 'b d n -> b n d')

        x = self.project_in(x)

        if is_multiheaded:
            x = rearrange(x, 'b n (h d) -> (b h) n d', h=heads)

        quantize, embed_ind = self._codebook(x, svq_temp, node_mask)

        # if self.training:
        if self.vq_loss_weight > 0 and self.learnable_codebook:
            dist = self.manifold.dist(
                quantize,
                x.detach()
            )
            dist = dist ** 2
            vq_loss1 = (dist * node_mask).sum() / node_mask.sum()
            vq_loss['vq_loss'] = vq_loss1
            vq_loss['loss'] = vq_loss['loss'] + vq_loss1 * self.vq_loss_weight
        else:
            vq_loss['vq_loss'] = 0

        if self.commitment_weight > 0:
            dist2 = self.manifold.dist(
                quantize.detach(),
                x
            )
            dist2 = dist2 ** 2
            commit_loss = (dist2 * node_mask).sum() / node_mask.sum()
            vq_loss['commit_loss'] = commit_loss
            vq_loss['loss'] = vq_loss['loss'] + commit_loss * self.commitment_weight


        if self.orthogonal_reg_weight > 0:
            codebook = self.codebook

            if self.orthogonal_reg_active_codes_only:
                # only calculate orthogonal loss for the activated codes for this batch
                unique_code_ids = torch.unique(embed_ind)
                codebook = codebook[unique_code_ids]

            num_codes = codebook.shape[0]
            if exists(self.orthogonal_reg_max_codes) and num_codes > self.orthogonal_reg_max_codes:
                rand_ids = torch.randperm(num_codes, device=device)[:self.orthogonal_reg_max_codes]
                codebook = codebook[rand_ids]

            orthogonal_reg_loss = orthgonal_loss_fn(codebook)
            vq_loss['orthogonal_reg_loss'] = orthogonal_reg_loss
            vq_loss['loss'] = vq_loss['loss'] + orthogonal_reg_loss * self.orthogonal_reg_weight

        if self.training:
            quantize = x + (quantize - x).detach()  # allows `z`-part to be trainable while `z_q`-part is un-trainable. `z_q` is updated by the EMA.

        if is_multiheaded:
            quantize = rearrange(quantize, '(b h) n d -> b n (h d)', h=heads)
            embed_ind = rearrange(embed_ind, '(b h) n -> b n h', h=heads)

        quantize = self.project_out(quantize)
        # 打印codebook embed的norm
        embed_norm = torch.norm(self.codebook, p=2, dim=-1)
        # 用loss传出来
        vq_loss['codebook_embed_norm'] = embed_norm
        
        if need_transpose:
            quantize = rearrange(quantize, 'b n d -> b d n')

        if self.accept_image_fmap:
            quantize = rearrange(quantize, 'b (h w) c -> b c h w', h=height, w=width)
            embed_ind = rearrange(embed_ind, 'b (h w) ... -> b h w ...', h=height, w=width)

        return quantize, embed_ind, vq_loss, self._codebook.perplexity


if __name__ == '__main__':
    torch.manual_seed(0)

    B, N, D = 1024, 32, 128
    x = torch.rand((B, N, D))

    vq = VectorQuantize(dim=D, codebook_size=512, use_hyperbolic=True)
    # vq = VectorQuantize(dim=D, codebook_size=512)

    quantize, vq_ind, vq_loss, perplexity = vq(x)
    '''
    quantize,  # 量化后的特征表示，形状与输入x相同（B, N, D）
    vq_ind,    # 量化索引矩阵，表示每个输入向量对应的码本索引（B, N）
    vq_loss,   # 包含三部分损失的字典：总损失/commitment损失/正交正则损失
    perplexity # 码本使用的困惑度（信息熵指标，反映码本利用率）
    '''
    print(vq_ind[0])  # `vq_ind` is a set of codebook indices; e.g., 87 denotes the 88-th code in the codebook which can be accessed by `vq.codebook[87]`.

    # you can fetch the codebook weight by `vq.codebook`
    print('vq.codebook.shape:', vq.codebook.shape)  # (codebook_size, dim)

