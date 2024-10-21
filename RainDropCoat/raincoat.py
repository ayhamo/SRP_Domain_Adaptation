import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_metric_learning import losses
import torch.fft

from Raincoat.algorithms.loss import SinkhornDistance
from raindrop import Raindrop_v2  # Import Raindrop

import warnings
warnings.filterwarnings("ignore", message="enable_nested_tensor is True, but self.use_nested_tensor is False because")
warnings.filterwarnings("ignore", message="Torch was not compiled with flash attention")

class Algorithm(torch.nn.Module):

    def __init__(self, configs):
        super(Algorithm, self).__init__()
        self.configs = configs
        self.cross_entropy = nn.CrossEntropyLoss()

    def update(self, *args, **kwargs):
        raise NotImplementedError

# Fractional Fourier Transform, better perfomance
# http://yoksis.bilkent.edu.tr/pdf/files/16189.pdf

class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, fraction_order):
        super(SpectralConv1d, self).__init__()
        """
        1D Fractional Fourier layer. It does FrFT, linear transform, and Inverse FrFT.    
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.fraction_order = fraction_order  # Fractional order for FrFT

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))
        self.pi = torch.acos(torch.zeros(1)).item() * 2

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input, weights)

    def frft(self, x, a):
        """
        Perform the Fractional Fourier Transform (FrFT) on the input tensor x with order a.
        """
        N = x.shape[-1]
        k = torch.arange(0, N, device=x.device)
        exp_term = torch.exp(-1j * self.pi * a * k**2 / N)
        x_ft = torch.fft.fft(x)
        x_frft = torch.fft.ifft(x_ft * exp_term)
        return x_frft

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fractional Fourier coefficients up to factor of e^(- something constant)
        x = torch.cos(x)
        x_frft = self.frft(x, self.fraction_order)
        out_frft = torch.zeros(batchsize, self.out_channels, x_frft.size(-1), device=x.device, dtype=torch.cfloat)
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
        self.fraction_order = configs.fraction_order

        self.freq_feature = SpectralConv1d(self.width, self.width, self.modes1, self.fraction_order)  # Frequency Feature Encoder
        self.bn_freq = nn.BatchNorm1d(configs.fourier_modes*2)   # It doubles because frequency features contain both amplitude and phase
        self.avg = nn.Conv1d(self.width, 1, kernel_size=3 ,
                  stride=configs.stride, bias=False, padding=(3 // 2))

        d_inp = configs.input_channels # number of input features, like WISDM x,y,z (3)
        d_ob = 40
        d_model =  d_inp * d_ob # number of expected model input features
        nhead = 4 # number of heads in multihead-attention
        nhid = 2 * d_model # configs.mid_channels * 4 is as per CNN configs, ask here?
        # ^ dimension of feedforward network model
        nlayers = 3

        # Initialize Raindrop_v2
        self.raindrop = Raindrop_v2(
            d_inp= d_inp,
            d_model= d_model,
            nhead= nhead,
            nhid= nhid,
            nlayers= nlayers,
            dropout= configs.dropout,
            max_len= configs.sequence_len,
            d_static=0,  # No static features yet
            MAX= 100,
            aggreg='mean',
            n_classes=configs.num_classes, # We don't need this for feature extraction but it's required by the model
            global_structure= torch.ones(d_inp, d_inp),
            sensor_wise_mask=False,  # Not using sensor-wise masks in this integration
            static=False,  # No static features
            device= device
        )
        
        self.projection = nn.Sequential(
            nn.Linear(d_model + 16, configs.sequence_len), #scale it down to seq length
            nn.ReLU(),
            nn.BatchNorm1d(configs.sequence_len)
        )

    def forward(self, x):
        ef, out_ft = self.freq_feature(x)
        ef = F.relu(self.bn_freq(self.avg(ef).squeeze()))

        # --- Preprocessing for Raindrop ---
        batch_size, channels, seq_len = x.shape

        # Permute to (seq_len, batch_size, channels)
        x_raindrop = x.permute(2, 0, 1) 

        # Create dummy static data
        dummy_static = None  # Or create a zero tensor if Raindrop requires it
        
        # Create dummy times
        times = torch.arange(seq_len, device=x.device).unsqueeze(1).repeat(1, batch_size).float()
        # Create lengths ( assuming no missing data for now)
        lengths = torch.full((batch_size,), seq_len, device=x.device)

        # Create dummy mask, since dataset has no missing values. concatenate with x_raindrop
        dummy_mask = torch.zeros_like(x_raindrop)
        x_raindrop = torch.cat([x_raindrop, dummy_mask], dim = 2)


        # --- Raindrop Feature Extraction ---
        # this now gives us [32, 136 [d_model(3 * 40) + 16] ], it's impossible to get 128 becuase of WISDM 3 channels
        # so we project it after down to 128 with a simple nn model
        et, _, _ = self.raindrop(x_raindrop, dummy_static, times, lengths)

        #print(f"et shape at encoder before projection: {et.shape}")

        et = self.projection(et)

        f = torch.concat([ef, et], -1)
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

        self.fraction_order = configs.fraction_order
        self.pi = torch.acos(torch.zeros(1)).item() * 2

    def inverse_frft(self, x, a):
        """
        Perform the inverse Fractional Fourier Transform (FrFT) on the input tensor x with order a.
        """
        N = x.shape[-1]
        k = torch.arange(0, N, device=x.device)
        exp_term = torch.exp(1j * self.pi * a * k**2 / N)  # Note the positive sign for inverse
        x_ft = torch.fft.fft(x)
        x_inv_frft = torch.fft.ifft(x_ft * exp_term)
        return x_inv_frft
        
    def forward(self, f, out_ft):
        # Reconstruct time series by using low frequency features from FrFT
        x_low_complex = self.inverse_frft(out_ft, self.fraction_order)

        amplitude = x_low_complex.abs()
        phase = x_low_complex.angle()

        x_low = torch.cat([amplitude, phase], dim=1)
        x_low = self.channel_proj(x_low)
        x_low = self.bn1(x_low)

        et = f[:, self.modes * 2:]

        # Reconstruct time series by using time features for high frequency patterns
        x_high = F.relu(self.bn2(self.convT(et.unsqueeze(2)).permute(0, 2, 1)))

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
        self.feature_extractor = tf_encoder(configs, device).to(device)
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

    def align(self, src_x, src_y, trg_x):
        self.optimizer.zero_grad()

        # Encode both source and target features via our time-frequency feature encoder
        src_feat, out_s = self.feature_extractor(src_x)
        trg_feat, out_t = self.feature_extractor(trg_x)
        # Decode extracted features to time series
        src_recon = self.decoder(src_feat, out_s)
        trg_recon = self.decoder(trg_feat, out_t)

        # Compute reconstruction loss (added the 0.2 weight here as per paper)
        recons = 1e-4 * (self.recons(src_recon, src_x) + self.recons(trg_recon, trg_x))
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

        return {
            'Total_loss': total_loss.item(),
            'Reconstruction_loss': recons.item(),
            'Alignment_loss': sink_loss.item(),
            'Classification_loss': loss_cls.item()
            }


    def correct(self,src_x, src_y, trg_x):
        self.coptimizer.zero_grad()

        src_feat, out_s = self.feature_extractor(src_x)
        trg_feat, out_t = self.feature_extractor(trg_x)

        src_recon = self.decoder(src_feat, out_s)
        trg_recon = self.decoder(trg_feat, out_t)

        recons = 1e-4 * (self.recons(trg_recon, trg_x) + self.recons(src_recon, src_x))
        recons.backward()

        self.coptimizer.step()

        return {'Correct_reconstruction_loss': recons.item()}
