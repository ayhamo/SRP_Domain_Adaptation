import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_metric_learning import losses

from Raincoat.algorithms.loss import SinkhornDistance

from raindrop import Raindrop_v2

class Algorithm(torch.nn.Module):

    def __init__(self, configs):
        super(Algorithm, self).__init__()
        self.configs = configs
        self.cross_entropy = nn.CrossEntropyLoss()

    def update(self, *args, **kwargs):
        raise NotImplementedError

# Fractional Fourier Transform, better perfomance
# http://yoksis.bilkent.edu.tr/pdf/files/16189.pdf
def frft(x, a):
    N = x.shape[-1]
    k = torch.arange(N, device=x.device)
    exp_term = torch.exp(-1j * torch.pi * a * k**2 / N)
    x = x * exp_term
    x_ft = torch.fft.fft(x)
    x_ft = x_ft * exp_term
    return torch.fft.ifft(x_ft)

class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, fl=128):
        super(SpectralConv1d, self).__init__()

        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  #Number of Fourier modes to multiply, at most floor(N/2) + 1

        self.scale = (1 / (in_channels*out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))
        self.pi = torch.acos(torch.zeros(1)).item() * 2

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute FrFT coefficients (only part that changed in the code)
        x = torch.cos(x)

        # a value (0.8) is a hyper paramter, grid search gave it as best
        x_frft = frft(x, 0.8)
        out_frft = torch.zeros(batchsize, self.out_channels, x.size(-1), device=x.device, dtype=torch.cfloat)
        out_frft[:, :, :self.modes1] = self.compl_mul1d(x_frft[:, :, :self.modes1], self.weights1)
        r = out_frft[:, :, :self.modes1].abs()
        p = out_frft[:, :, :self.modes1].angle()
        return torch.concat([r, p], -1), out_frft

class tf_encoder(nn.Module):
    def __init__(self, configs, device):
        super(tf_encoder, self).__init__()
        self.modes1 = configs.fourier_modes   # Number of low-frequency modes to keep
        self.width = configs.input_channels
        self.length =  configs.sequence_len

        self.freq_feature = SpectralConv1d(self.width, self.width, self.modes1,self.length)  # Frequency Feature Encoder
        self.bn_freq = nn.BatchNorm1d(configs.fourier_modes*2)   # It doubles because frequency features contain both amplitude and phase
        self.avg = nn.Conv1d(self.width, 1, kernel_size=3 ,
                  stride=configs.stride, bias=False, padding=(3 // 2))

        # No longer need CNN, but we will use raindrop
        # Integrate RAINDROP
        self.raindrop = Raindrop_v2(
            d_inp=configs.input_channels,             # Number of input sensor channels (3 for WISDM)
            d_model=configs.final_out_channels,        # Should match CNN output dimension (128 for WISDM) 
            nhead=4,                                  # Number of attention heads (can be adjusted)
            nhid=2 * configs.final_out_channels,       # Hidden dimension in transformer (256 for WISDM)
            nlayers=2,                                 # Number of transformer layers (can be adjusted)
            dropout=configs.dropout,                  # Dropout rate (0.5 for WISDM)
            max_len=configs.sequence_len,             # Maximum sequence length (128 for WISDM)
            d_static=0,                               # No static features in WISDM (for now)
            MAX=100,                                 # Positional encoding parameter (can be adjusted)
            perc=0.5,                                 # Percentage of edges to prune in RAINDROP (can be adjusted)
            aggreg='mean',                           # Aggregation method for sensor embeddings ('mean' is common)
            n_classes=configs.num_classes,            # Number of output classes (6 for WISDM)
            global_structure=None,                   # Initialized later in RAINCOAT
            sensor_wise_mask=False,                  # Not using sensor-wise masks in this integration
            static=False                              # No static features in WISDM (for now)
            device = device
        )


    def forward(self, src, static, times, lengths):
        ef, out_ft = self.freq_feature(src)
        ef = F.relu(self.bn_freq(self.avg(ef).squeeze()))

        # Reshape src for RAINDROP
        src = src.permute(2, 0, 1) # Now shape is [sequence_length, batch_size, input_channels]

        # Use RAINDROP for time-domain feature extraction
        et, _, _ = self.raindrop(src, static, times, lengths)


        f = torch.concat([ef,et],-1)
        return F.normalize(f), out_ft


class tf_decoder(nn.Module):
    def __init__(self, configs):
        super(tf_decoder, self).__init__()
        self.input_channels, self.sequence_len = configs.input_channels, configs.sequence_len
        self.bn1 = nn.BatchNorm1d(self.input_channels,self.sequence_len)
        self.bn2 = nn.BatchNorm1d(self.input_channels,self.sequence_len)
        self.convT = torch.nn.ConvTranspose1d(configs.final_out_channels, self.sequence_len, self.input_channels, stride=1)
        self.modes = configs.fourier_modes

        self.channel_proj = nn.Conv1d(2 * self.input_channels, self.input_channels, kernel_size=1)

    def forward(self, f, out_ft, timestamps):
        # Reconstruct time series using low-frequency FrFT features
        a= -0.8
        x_low_mag = torch.abs(frft(out_ft, a))  # Magnitude
        x_low_phase = torch.angle(frft(out_ft, a))  # Phase
        x_low = torch.cat([x_low_mag, x_low_phase], dim=1)  # Concatenate
        x_low = self.channel_proj(x_low)  # Project back to num_channels

        et = f[:, self.modes * 2:]

        # Interpolate time features at the given timestamps 
        et_interp = F.interpolate(et.unsqueeze(1), size=len(timestamps), mode='linear', align_corners=True).squeeze(1)

        # Reconstruct using interpolated time features
        x_high = F.relu(self.bn2(self.convT(et_interp.unsqueeze(2)).permute(0, 2, 1)))  

        return x_low + x_high

class classifier(nn.Module):
    def __init__(self, configs):
        super(classifier, self).__init__()
        model_output_dim = configs.out_dim
        self.logits = nn.Linear(model_output_dim, configs.num_classes, bias=False)
        self.tmp= 0.1

    def forward(self, x):
        predictions = self.logits(x)/self.tmp
        return predictions


class RAINCOAT(Algorithm):
    def __init__(self, configs, hparams, device):
        super(RAINCOAT, self).__init__(configs)
        self.feature_extractor = tf_encoder(configs,device).to(device)
        self.decoder = tf_decoder(configs).to(device)
        self.classifier = classifier(configs).to(device)

        self.optimizer = torch.optim.AdamW(
            list(self.feature_extractor.parameters()) + \
                list(self.decoder.parameters())+\
                list(self.classifier.parameters()),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )

        # Super-Convergence: Very Fast Training of Neural Networks Using Large Learning Rates".
        # https://arxiv.org/abs/1708.07120

        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            cycle_momentum = True,
            max_lr= 1e-2,
            steps_per_epoch=hparams["scheduler_steps"],
            epochs=hparams["num_epochs"]
        )

        self.coptimizer = torch.optim.AdamW(
            list(self.feature_extractor.parameters())+list(self.decoder.parameters()),
            lr=1*hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        
        self.coscheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.coptimizer,
            cycle_momentum = True,
            max_lr= 1e-2,
            steps_per_epoch=hparams["coscheduler_steps"],
            epochs=hparams["corr_epochs"]
        )

        self.hparams = hparams
        self.recons = nn.L1Loss(reduction='sum').to(device)
        self.pi = torch.acos(torch.zeros(1)).item() * 2
        self.loss_func = losses.ContrastiveLoss(pos_margin=0.5)
        self.sink = SinkhornDistance(eps=1e-3, max_iter=1000, reduction='sum', device=device)

    def align(self, src_x, src_y, src_times, src_lengths, trg_x, trg_times, trg_lengths): 
        self.optimizer.zero_grad()

        # Encode both source and target features via our time-frequency feature encoder
        src_feat, out_s = self.feature_extractor(src_x, None, src_times, src_lengths)
        trg_feat, out_t = self.feature_extractor(trg_x, None, trg_times, trg_lengths)

        # Decode extracted features to time series
        src_recon = self.decoder(src_feat, out_s, src_times)  
        trg_recon = self.decoder(trg_feat, out_t, trg_times) 


        # Calculate weights for the reconstruction loss
        src_weights = (src_times > 0).float()
        trg_weights = (trg_times > 0).float() 

        # Compute reconstruction loss (added the 0.2 weight here as per paper)
        recons = 1e-4 * (self.weighted_l1_loss(src_recon, src_x, src_weights) + 
                         self.weighted_l1_loss(trg_recon, trg_x, trg_weights))
        recons.backward(retain_graph=True)

        # Compute alignment loss
        dr, _, _ = self.sink(src_feat, trg_feat)
        sink_loss = dr
        sink_loss.backward(retain_graph=True)

        # Compute classification loss
        src_pred = self.classifier(src_feat)
        loss_cls = self.cross_entropy(src_pred, src_y)
        loss_cls.backward(retain_graph=True)

        # Compute weights
        a, b, c = 1, 1, 0.2
        total = a + b + c
        lambda1 = a / total
        lambda2 = b / total
        lambda3 = c / total

        # Compute total loss with weights
        total_loss = lambda1 * recons + lambda2 * sink_loss + lambda3 * loss_cls
        
        self.optimizer.step()
        self.scheduler.step()

        return {
            'Total_loss': total_loss.item(),
            'Reconstruction_loss': recons.item(),
            'Alignment_loss': sink_loss.item(),
            'Classification_loss': loss_cls.item()
            }


    def correct(self, src_x, src_y, src_times, src_lengths, trg_x, trg_times, trg_lengths):
        self.coptimizer.zero_grad()

        src_feat, out_s = self.feature_extractor(src_x, None, src_times, src_lengths)
        trg_feat, out_t = self.feature_extractor(trg_x, None, trg_times, trg_lengths)

        src_recon = self.decoder(src_feat, out_s, src_times)  
        trg_recon = self.decoder(trg_feat, out_t, trg_times) 

        # Calculate weights for the reconstruction loss
        src_weights = (src_times > 0).float()
        trg_weights = (trg_times > 0).float()

        # Compute weighted reconstruction loss
        recons = 1e-4 * (self.weighted_l1_loss(trg_recon, trg_x, trg_weights) + 
                         self.weighted_l1_loss(src_recon, src_x, src_weights)) 

        recons.backward()

        self.coptimizer.step()
        self.scheduler.step()

        return {'Correct_reconstruction_loss': recons.item()}
    
    def weighted_l1_loss(input, target, weights):
        r"""
        Calculates the weighted L1 loss between input and target.

        Args:
            input (torch.Tensor): The input tensor.
            target (torch.Tensor): The target tensor.
            weights (torch.Tensor): Weights for each time point. Should have the same shape as input and target.
        
        Returns:
            torch.Tensor: The weighted L1 loss.
        """
        return torch.sum(weights * torch.abs(input - target))
