#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
External subclass that fixes the public repository's hard-coded VQ-VAE path
without modifying upstream source files.

Also patches PyTorch>=2.6 + old PyTorch Lightning checkpoint loading behavior:
torch.load() now defaults to weights_only=True, while legacy Lightning
load_from_checkpoint() expects the older full-checkpoint behavior.
"""
from utils.lightning_compat import patch_torchmetrics_for_legacy_lightning

patch_torchmetrics_for_legacy_lightning()

import copy
import pytorch_lightning as pl
import torch
import torch.nn as nn

from skygpt.vqvae import VQVAE
from skygpt.gpt import (
    AddBroadcastPosEmbed,
    AttentionStack,
    LayerNorm,
    PhyCell,
    SkyGPT as RepoSkyGPT,
    resnet34,
)


def _patch_legacy_lightning_checkpoint_loading():
    try:
        import pytorch_lightning.core.saving as pl_saving
    except Exception:
        pl_saving = None

    try:
        import pytorch_lightning.utilities.cloud_io as pl_cloud_io
    except Exception:
        pl_cloud_io = None

    def _compat_load(f, map_location=None):
        return torch.load(f, map_location=map_location, weights_only=False)

    if pl_saving is not None:
        pl_saving.pl_load = _compat_load
    if pl_cloud_io is not None:
        pl_cloud_io.load = _compat_load


class ExternalSkyGPT(RepoSkyGPT):
    def __init__(self, args):
        pl.LightningModule.__init__(self)
        self.args = args
        if not hasattr(args, "vqvae") or not args.vqvae:
            raise ValueError("args.vqvae is required and must point to your trained external VQ-VAE checkpoint.")

        _patch_legacy_lightning_checkpoint_loading()
        self.vqvae = VQVAE.load_from_checkpoint(args.vqvae, map_location="cpu")
        for p in self.vqvae.parameters():
            p.requires_grad = False
        self.vqvae.codebook._need_init = False
        self.vqvae.eval()

        self.use_frame_cond = args.n_cond_frames > 0
        if self.use_frame_cond:
            frame_cond_shape = (
                args.n_cond_frames,
                args.resolution // 4,
                args.resolution // 4,
                240,
            )
            self.resnet = resnet34(1, (1, 4, 4), resnet_dim=240)
            self.cond_pos_embd = AddBroadcastPosEmbed(
                shape=frame_cond_shape[:-1],
                embd_dim=frame_cond_shape[-1],
            )
        else:
            frame_cond_shape = None

        self.shape = self.vqvae.latent_shape
        self.fc_in = nn.Linear(self.vqvae.embedding_dim, args.hidden_dim, bias=False)
        self.fc_in.weight.data.normal_(std=0.02)

        self.attn_stack = AttentionStack(
            self.shape,
            args.hidden_dim,
            args.heads,
            args.layers,
            args.dropout,
            args.attn_type,
            args.attn_dropout,
            args.class_cond_dim,
            frame_cond_shape,
        )

        self.norm = LayerNorm(args.hidden_dim, args.class_cond_dim)
        self.fc_out = nn.Linear(args.hidden_dim, self.vqvae.n_codes, bias=False)
        self.fc_out.weight.data.copy_(torch.zeros(self.vqvae.n_codes, args.hidden_dim))

        self.frame_cond_cache = None
        # Lightning 1.4.x always tries to inspect the __init__ frame inside
        # save_hyperparameters(), even when a dict/Namespace is passed in.
        # Because this wrapper bypasses the upstream constructor path, that
        # reflection fails here. Set hparams directly instead.
        self._set_hparams(vars(args))
        self._hparams_initial = copy.deepcopy(self.hparams)

        self.kernel_size = 7
        f_hidden_dims = self.kernel_size ** 2
        phycell_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.phycell = PhyCell(
            input_shape=(32, 32),
            input_dim=args.hidden_dim,
            F_hidden_dims=f_hidden_dims,
            n_layers=1,
            kernel_size=(self.kernel_size, self.kernel_size),
            heads=args.heads,
            device=phycell_device,
        )

        self.constraint_criterion = nn.MSELoss()
        self.constraints = torch.zeros((f_hidden_dims, self.kernel_size, self.kernel_size))
        for i in range(self.kernel_size):
            for j in range(self.kernel_size):
                self.constraints[i * self.kernel_size + j, i, j] = 1
