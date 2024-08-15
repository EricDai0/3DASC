
from pprint import pprint
#from evaluation.evaluation_metrics import compute_all_metrics, EMD_CD

from metrics.evaluation_metrics import compute_all_metrics, EMD_CD

import torch.nn as nn
import torch.utils.data

import argparse
from torch.distributions import Normal
from utils.file_utils import *
from utils.visualize import *
from model.pvcnn_completion import PVCNN2Base
from torch.backends import cudnn

import torch.nn.functional as F

from models.pointnet import PointNetCls, feature_transform_regularizer
from models.pointnet2 import PointNet2ClsMsg
from models.dgcnn import DGCNN
from models.pointcnn import PointCNNCls

from model import DGCNN, PointNetCls, PointNet2ClsSsg, PointConvDensityClsSsg,GDANET, RPC, feature_transform_reguliarzer

from datasets.shapenet_data_pc import ShapeNet15kPointClouds
from datasets.shapenet_data_sv import *
'''
models

A = np.arange(100)

arr = np.random.choice(100, 20, replace=False)

S2 = A[list(set(range(100)) - set(arr))]

'''

cudnn.benchmark = False
cudnn.deterministic = True
torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)
random.seed(0)

class SORDefense(nn.Module):
    """Statistical outlier removal as defense.
    """

    def __init__(self, k=2, alpha=1.1):
        """SOR defense.

        Args:
            k (int, optional): kNN. Defaults to 2.
            alpha (float, optional): \miu + \alpha * std. Defaults to 1.1.
        """
        super(SORDefense, self).__init__()

        self.k = k
        self.alpha = alpha

    def outlier_removal(self, x):
        """Removes large kNN distance points.

        Args:
            x (torch.FloatTensor): batch input pc, [B, K, 3]

        Returns:
            torch.FloatTensor: pc after outlier removal, [B, N, 3]
        """
        pc = x.clone().detach().double()
        B, K = pc.shape[:2]
        pc = pc.transpose(2, 1)  # [B, 3, K]
        inner = -2. * torch.matmul(pc.transpose(2, 1), pc)  # [B, K, K]
        xx = torch.sum(pc ** 2, dim=1, keepdim=True)  # [B, 1, K]
        dist = xx + inner + xx.transpose(2, 1)  # [B, K, K]
        assert dist.min().item() >= -1e-6
        # the min is self so we take top (k + 1)
        neg_value, _ = (-dist).topk(k=self.k + 1, dim=-1)  # [B, K, k + 1]
        value = -(neg_value[..., 1:])  # [B, K, k]
        value = torch.mean(value, dim=-1)  # [B, K]
        mean = torch.mean(value, dim=-1)  # [B]
        std = torch.std(value, dim=-1)  # [B]
        threshold = mean + self.alpha * std  # [B]
        bool_mask = (value <= threshold[:, None])  # [B, K]
        sel_pc = [x[i][bool_mask[i]] for i in range(B)]
        return sel_pc

    def forward(self, x):
        with torch.no_grad():
            x = self.outlier_removal(x)
        return x
    
    
def sor(pc,defense_module):
        all_defend_pc = []
        batch_pc = pc
        batch_pc = batch_pc.float().cuda()
        print(batch_pc)
        defend_batch_pc = defense_module(batch_pc)
        # sor processed results have different number of points in each
        if isinstance(defend_batch_pc, list) or \
                isinstance(defend_batch_pc, tuple):
            for pc in defend_batch_pc:
                pc = pc[:2048,:].unsqueeze(0)
                all_defend_pc.append(pc)
                print(pc.shape)
        else:
            defend_batch_pc = defend_batch_pc.\
                detach().cpu().numpy().astype(np.float32)
            defend_batch_pc = [pc for pc in defend_batch_pc]
        all_defend_pc = torch.cat(all_defend_pc,dim=0)

        return all_defend_pc
    
class ClipPointsLinf(nn.Module):

    def __init__(self, budget):
        """Clip point cloud with a given l_inf budget.

        Args:
            budget (float): perturbation budget
        """
        super(ClipPointsLinf, self).__init__()

        self.budget = budget

    def forward(self, pc, ori_pc):
        """Clipping every point in a point cloud.

        Args:
            pc (torch.FloatTensor): batch pc, [B, 3, K]
            ori_pc (torch.FloatTensor): original point cloud
        """
        with torch.no_grad():
            diff = pc - ori_pc  # [B, 3, K]
            norm = torch.sum(diff ** 2, dim=1) ** 0.5  # [B, K]
            scale_factor = self.budget / (norm + 1e-9)  # [B, K]
            scale_factor = torch.clamp(scale_factor, max=1.)  # [B, K]
            diff = diff * scale_factor[:, None, :]
            pc = ori_pc + diff
        return pc

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

def _jacobian(model, input, tar_idx, model_index):
    
    def atanh(x):
        return 0.5*torch.log((1+x)/(1-x))   
    
    def f(outputs, labels):
        one_hot_labels = torch.eye(len(outputs[0]))[labels].to(output.device)

        i, _ = torch.max((1-one_hot_labels)*outputs, dim=1) # get the second largest logit
        j = torch.masked_select(outputs, one_hot_labels.bool()) # get the largest logit

        #if self.targeted:
            #return torch.clamp((i-j), min=-self.kappa)
        #else:
        return torch.clamp((j-i), min=0)

     
    with torch.enable_grad():   

        w = input.detach().requires_grad_(True)

        #output = model(w)

        closs = nn.CrossEntropyLoss()

        if model_index == 0:
            logits, _, _ = model(w)
        else:
            logits = model(w)
                               
        log_probs = F.log_softmax(logits, dim=-1)
        #tar_label = get_target_label(logits, label, device)
        #selected = log_probs[range(len(logits)), tar_idx]
        loss = F.cross_entropy(logits, tar_idx)
        gradient = torch.autograd.grad(loss, w)[0]


    return gradient


def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    KL divergence between normal distributions parameterized by mean and log-variance.
    """
    return 0.5 * (-1.0 + logvar2 - logvar1 + torch.exp(logvar1 - logvar2)
                + (mean1 - mean2)**2 * torch.exp(-logvar2))

def discretized_gaussian_log_likelihood(x, *, means, log_scales):
    # Assumes data is integers [0, 1]
    assert x.shape == means.shape == log_scales.shape
    px0 = Normal(torch.zeros_like(means), torch.ones_like(log_scales))

    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 0.5)
    cdf_plus = px0.cdf(plus_in)
    min_in = inv_stdv * (centered_x - .5)
    cdf_min = px0.cdf(min_in)
    log_cdf_plus = torch.log(torch.max(cdf_plus, torch.ones_like(cdf_plus)*1e-12))
    log_one_minus_cdf_min = torch.log(torch.max(1. - cdf_min,  torch.ones_like(cdf_min)*1e-12))
    cdf_delta = cdf_plus - cdf_min

    log_probs = torch.where(
    x < 0.001, log_cdf_plus,
    torch.where(x > 0.999, log_one_minus_cdf_min,
             torch.log(torch.max(cdf_delta, torch.ones_like(cdf_delta)*1e-12))))
    assert log_probs.shape == x.shape
    return log_probs



class GaussianDiffusion:
    def __init__(self, betas, loss_type, model_mean_type, model_var_type, sv_points, args):
        self.args = args
        self.loss_type = loss_type
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        assert isinstance(betas, np.ndarray)
        self.np_betas = betas = betas.astype(np.float64)  # computations here in float64 for accuracy
        assert (betas > 0).all() and (betas <= 1).all()
        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.sv_points = sv_points
        # initialize twice the actual length so we can keep running for eval
        # betas = np.concatenate([betas, np.full_like(betas[:int(0.2*len(betas))], betas[-1])])

        alphas = 1. - betas
        alphas_cumprod = torch.from_numpy(np.cumprod(alphas, axis=0)).float()
        alphas_cumprod_prev = torch.from_numpy(np.append(1., alphas_cumprod[:-1])).float()

        self.betas = torch.from_numpy(betas).float()
        self.alphas_cumprod = alphas_cumprod.float()
        self.alphas_cumprod_prev = alphas_cumprod_prev.float()

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod).float()
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod).float()
        self.log_one_minus_alphas_cumprod = torch.log(1. - alphas_cumprod).float()
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1. / alphas_cumprod).float()
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1. / alphas_cumprod - 1).float()

        betas = torch.from_numpy(betas).float()
        alphas = torch.from_numpy(alphas).float()
        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.posterior_variance = posterior_variance
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.posterior_log_variance_clipped = torch.log(torch.max(posterior_variance, 1e-20 * torch.ones_like(posterior_variance)))
        self.posterior_mean_coef1 = betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.posterior_mean_coef2 = (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod)

    @staticmethod
    def _extract(a, t, x_shape):
        """
        Extract some coefficients at specified timesteps,
        then reshape to [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
        """
        bs, = t.shape
        assert x_shape[0] == bs
        out = torch.gather(a, 0, t)
        assert out.shape == torch.Size([bs])
        return torch.reshape(out, [bs] + ((len(x_shape) - 1) * [1]))



    def q_mean_variance(self, x_start, t):
        mean = self._extract(self.sqrt_alphas_cumprod.to(x_start.device), t, x_start.shape) * x_start
        variance = self._extract(1. - self.alphas_cumprod.to(x_start.device), t, x_start.shape)
        log_variance = self._extract(self.log_one_minus_alphas_cumprod.to(x_start.device), t, x_start.shape)
        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the data (t == 0 means diffused for 1 step)
        """
        if noise is None:
            noise = torch.randn(x_start.shape, device=x_start.device)
        assert noise.shape == x_start.shape
        return (
                self._extract(self.sqrt_alphas_cumprod.to(x_start.device), t, x_start.shape) * x_start +
                self._extract(self.sqrt_one_minus_alphas_cumprod.to(x_start.device), t, x_start.shape) * noise
        )


    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior q(x_{t-1} | x_t, x_0)
        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
                self._extract(self.posterior_mean_coef1.to(x_start.device), t, x_t.shape) * x_start +
                self._extract(self.posterior_mean_coef2.to(x_start.device), t, x_t.shape) * x_t
        )
        posterior_variance = self._extract(self.posterior_variance.to(x_start.device), t, x_t.shape)
        posterior_log_variance_clipped = self._extract(self.posterior_log_variance_clipped.to(x_start.device), t, x_t.shape)
        assert (posterior_mean.shape[0] == posterior_variance.shape[0] == posterior_log_variance_clipped.shape[0] ==
                x_start.shape[0])
        return posterior_mean, posterior_variance, posterior_log_variance_clipped


    def p_mean_variance(self, denoise_fn, data, t, clip_denoised: bool, return_pred_xstart: bool):

        model_output = denoise_fn(data, t)[:,:,self.sv_points:]


        if self.model_var_type in ['fixedsmall', 'fixedlarge']:
            # below: only log_variance is used in the KL computations
            model_variance, model_log_variance = {
                # for fixedlarge, we set the initial (log-)variance like so to get a better decoder log likelihood
                'fixedlarge': (self.betas.to(data.device),
                               torch.log(torch.cat([self.posterior_variance[1:2], self.betas[1:]])).to(data.device)),
                'fixedsmall': (self.posterior_variance.to(data.device), self.posterior_log_variance_clipped.to(data.device)),
            }[self.model_var_type]
            model_variance = self._extract(model_variance, t, data.shape) * torch.ones_like(model_output)
            model_log_variance = self._extract(model_log_variance, t, data.shape) * torch.ones_like(model_output)
        else:
            raise NotImplementedError(self.model_var_type)

        if self.model_mean_type == 'eps':
            x_recon = self._predict_xstart_from_eps(data[:,:,self.sv_points:], t=t, eps=model_output)


            model_mean, _, _ = self.q_posterior_mean_variance(x_start=x_recon, x_t=data[:,:,self.sv_points:], t=t)
        else:
            raise NotImplementedError(self.loss_type)


        assert model_mean.shape == x_recon.shape
        assert model_variance.shape == model_log_variance.shape
        if return_pred_xstart:
            return model_mean, model_variance, model_log_variance, x_recon
        else:
            return model_mean, model_variance, model_log_variance

    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
                self._extract(self.sqrt_recip_alphas_cumprod.to(x_t.device), t, x_t.shape) * x_t -
                self._extract(self.sqrt_recipm1_alphas_cumprod.to(x_t.device), t, x_t.shape) * eps
        )

    ''' samples '''

    def p_sample(self, denoise_fn, data, t, noise_fn, clip_denoised=False, return_pred_xstart=False):
        """
        Sample from the model
        """
        model_mean, model_variance, model_log_variance, pred_xstart = self.p_mean_variance(denoise_fn, data=data, t=t, clip_denoised=clip_denoised,
                                                                 return_pred_xstart=True)
        noise = noise_fn(size=model_mean.shape, dtype=model_mean.dtype, device=model_mean.device)

        # no noise when t == 0
        nonzero_mask = torch.reshape(1 - (t == 0).float(), [data.shape[0]] + [1] * (len(model_mean.shape) - 1))

        sample = model_mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise
        sample = torch.cat([data[:, :, :self.sv_points], sample], dim=-1)
        
        return (sample, pred_xstart) if return_pred_xstart else sample, model_variance


    def p_sample_loop(self, partial_x, denoise_fn, shape, device,
                      noise_fn=torch.randn, clip_denoised=True, keep_running=False, adv_models=None, model_index=0):
        """
        Generate samples
        keep_running: True if we run 2 x num_timesteps, False if we just run num_timesteps

        """

        assert isinstance(shape, (tuple, list))
        
        total=0
        correct=0
        step=0
        decay=1.0
        
        alpha=2 / 255
        steps=5
        
        label = torch.tensor(shape[0]*[args.category_label]).to(device)
        
        clip_func = ClipPointsLinf(budget=0.16)
        img_t = torch.cat([partial_x, noise_fn(size=shape, dtype=torch.float, device=device)], dim=-1)
        
        pri_img = img_t.detach().requires_grad_(True)
        
        momentum = torch.zeros(img_t.shape[0], img_t.shape[1], 1860).detach().to(device)
        
        for k in range(1):
            print(img_t[:,:,shape[2]:].shape)
            for t in reversed(range(0, self.num_timesteps if not keep_running else len(self.betas))):
                t_ = torch.empty(shape[0], dtype=torch.int64, device=device).fill_(t)
                img_t, model_variance = self.p_sample(denoise_fn=denoise_fn, data=img_t,t=t_, noise_fn=noise_fn,
                                      clip_denoised=clip_denoised, return_pred_xstart=False)

                if(t >= 0 * self.num_timesteps and t <= 0.10 * self.num_timesteps):
                    #step += 1
                    #if step == 2:
                        gradient = 0

                        
                        temp = img_t.clone()
                        
                        momentum = torch.zeros(img_t.shape[0], img_t.shape[1], 1860).detach().to(device)
                        for _ in range(steps):
                            #step = 0
                            pc_adv = img_t[:,:,self.sv_points:]
                            arr = np.random.choice(1860, 1660, replace=False)
                            img_n =  torch.cat([img_t[:,:,:self.sv_points], pc_adv[:,:,arr]], dim=2)
                            with torch.enable_grad():
                                x_in = img_n.detach().requires_grad_(True)
                                logits1, _, trans_feat = adv_models[0](x_in)
                                logits2 = adv_models[1](x_in)
                                logits3 = adv_models[2](x_in)
                                
                                correct1 = label.eq(logits1.data.max(1)[1]).sum()

                                correct2 = label.eq(logits2.data.max(1)[1]).sum()

                                correct3 = label.eq(logits3.data.max(1)[1]).sum()
                                logits = (logits1*correct1+logits2*correct2+logits3*correct3)/(correct1 + correct2 + correct3)
                                log_probs = F.log_softmax(logits, dim=-1)
                                #tar_label = get_target_label(logits, label, device)
                                selected = log_probs[range(len(logits)), label]
                                
                                gradient = -torch.autograd.grad(selected.sum(), x_in)[0]
                                
                                momentum += gradient

                            
                            prior_set = set(range(200))
                            model_index = []
                            logit = [logits1,logits2,logits3]
                            for i in range(len(adv_models)):
                                adv_model=adv_models[i]

                                saliency = _jacobian(adv_model, x_in, label, i).squeeze() 

                                saliency = torch.sum(saliency, 1)

                                rates, indices = saliency.sort(descending=True) 
                                model_index.append(indices[:,:100])

                            for j in range(img_t.shape[0]):

                                index_set = set(model_index[0][j]) | set(model_index[1][j]) | set(model_index[2][j]) - prior_set
                                img_t[j, :, list(index_set)] = img_t[j, :, list(index_set)] +  0.05 * (1 - model_variance)[j, :, list(index_set)] * momentum[j, :, list(index_set)].float()
                                
                            img_t = clip_func(img_t, temp)

            
            model_index = 0
            gradient = 0

            if model_index == 0:
                output, _, _ = adv_models[0](img_t)
            else:
                output = adv_model[model_index](img_t)
            pred = output.data.max(1, keepdim=True)[1]
            correct = pred.eq(label.data.view_as(pred)).sum()
            total = shape[0]
            if correct == 0 : # early exit
                break
        model_index = 0
        pred_index = 0
        for i in range(len(adv_models)):
            adv_model = adv_models[i]
            if model_index == 0:
                output, _, _ = adv_model(img_t)
            else:
                output = adv_model(img_t)
            model_index += 1
            pred = output.data.max(1)[1]
            pred_index += (pred == label).long()
        print(pred_index)
        total = shape[0]
        correct = (pred_index != 0).sum()
        print("ASR:")
        print(correct/total)
        
        pred = torch.argmax(output, dim=1)  # [B]
        
        assert img_t[:,:,self.sv_points:].shape == shape
        return img_t[pred_index == 0], [pred_index == 0]

    def p_sample_loop_trajectory(self, denoise_fn, shape, device, freq,
                                 noise_fn=torch.randn,clip_denoised=True, keep_running=False):
        """
        Generate samples, returning intermediate images
        Useful for visualizing how denoised images evolve over time
        Args:
          repeat_noise_steps (int): Number of denoising timesteps in which the same noise
            is used across the batch. If >= 0, the initial noise is the same for all batch elemements.
        """
        assert isinstance(shape, (tuple, list))

        total_steps =  self.num_timesteps if not keep_running else len(self.betas)

        img_t = noise_fn(size=shape, dtype=torch.float, device=device)
        imgs = [img_t]
        for t in reversed(range(0,total_steps)):

            t_ = torch.empty(shape[0], dtype=torch.int64, device=device).fill_(t)
            img_t = self.p_sample(denoise_fn=denoise_fn, data=img_t, t=t_, noise_fn=noise_fn,
                                  clip_denoised=clip_denoised,
                                  return_pred_xstart=False)
            if t % freq == 0 or t == total_steps-1:
                imgs.append(img_t)

        assert imgs[-1].shape == shape
        return imgs

    '''losses'''

    def _vb_terms_bpd(self, denoise_fn, data_start, data_t, t, clip_denoised: bool, return_pred_xstart: bool):
        true_mean, _, true_log_variance_clipped = self.q_posterior_mean_variance(x_start=data_start[:,:,self.sv_points:], x_t=data_t[:,:,self.sv_points:], t=t)
        model_mean, _, model_log_variance, pred_xstart = self.p_mean_variance(
            denoise_fn, data=data_t, t=t, clip_denoised=clip_denoised, return_pred_xstart=True)

        kl = normal_kl(true_mean, true_log_variance_clipped, model_mean, model_log_variance)
        kl = kl.mean(dim=list(range(1, len(model_mean.shape)))) / np.log(2.)

        return (kl, pred_xstart) if return_pred_xstart else kl

    def p_losses(self, denoise_fn, data_start, t, noise=None):
        """
        Training loss calculation
        """
        B, D, N = data_start.shape
        assert t.shape == torch.Size([B])

        if noise is None:
            noise = torch.randn(data_start[:,:,self.sv_points:].shape, dtype=data_start.dtype, device=data_start.device)

        data_t = self.q_sample(x_start=data_start[:,:,self.sv_points:], t=t, noise=noise)

        if self.loss_type == 'mse':
            # predict the noise instead of x_start. seems to be weighted naturally like SNR
            eps_recon = denoise_fn(torch.cat([data_start[:,:,:self.sv_points], data_t], dim=-1), t)[:,:,self.sv_points:]

            losses = ((noise - eps_recon)**2).mean(dim=list(range(1, len(data_start.shape))))
        elif self.loss_type == 'kl':
            losses = self._vb_terms_bpd(
                denoise_fn=denoise_fn, data_start=data_start, data_t=data_t, t=t, clip_denoised=False,
                return_pred_xstart=False)
        else:
            raise NotImplementedError(self.loss_type)

        assert losses.shape == torch.Size([B])
        return losses

    '''debug'''

    def _prior_bpd(self, x_start):

        with torch.no_grad():
            B, T = x_start.shape[0], self.num_timesteps
            t_ = torch.empty(B, dtype=torch.int64, device=x_start.device).fill_(T-1)
            qt_mean, _, qt_log_variance = self.q_mean_variance(x_start, t=t_)
            kl_prior = normal_kl(mean1=qt_mean, logvar1=qt_log_variance,
                                 mean2=torch.tensor([0.]).to(qt_mean), logvar2=torch.tensor([0.]).to(qt_log_variance))
            assert kl_prior.shape == x_start.shape
            return kl_prior.mean(dim=list(range(1, len(kl_prior.shape)))) / np.log(2.)

    def calc_bpd_loop(self, denoise_fn, x_start, clip_denoised=True):

        with torch.no_grad():
            B, T = x_start.shape[0], self.num_timesteps

            vals_bt_, mse_bt_= torch.zeros([B, T], device=x_start.device), torch.zeros([B, T], device=x_start.device)
            for t in reversed(range(T)):

                t_b = torch.empty(B, dtype=torch.int64, device=x_start.device).fill_(t)
                # Calculate VLB term at the current timestep
                data_t = torch.cat([x_start[:, :, :self.sv_points], self.q_sample(x_start=x_start[:, :, self.sv_points:], t=t_b)], dim=-1)
                new_vals_b, pred_xstart = self._vb_terms_bpd(
                    denoise_fn, data_start=x_start, data_t=data_t, t=t_b,
                    clip_denoised=clip_denoised, return_pred_xstart=True)
                # MSE for progressive prediction loss
                assert pred_xstart.shape == x_start[:, :, self.sv_points:].shape
                new_mse_b = ((pred_xstart - x_start[:, :, self.sv_points:]) ** 2).mean(dim=list(range(1, len(pred_xstart.shape))))
                assert new_vals_b.shape == new_mse_b.shape ==  torch.Size([B])
                # Insert the calculated term into the tensor of all terms
                mask_bt = t_b[:, None]==torch.arange(T, device=t_b.device)[None, :].float()
                vals_bt_ = vals_bt_ * (~mask_bt) + new_vals_b[:, None] * mask_bt
                mse_bt_ = mse_bt_ * (~mask_bt) + new_mse_b[:, None] * mask_bt
                assert mask_bt.shape == vals_bt_.shape == vals_bt_.shape == torch.Size([B, T])

            prior_bpd_b = self._prior_bpd(x_start[:,:,self.sv_points:])
            total_bpd_b = vals_bt_.sum(dim=1) + prior_bpd_b
            assert vals_bt_.shape == mse_bt_.shape == torch.Size([B, T]) and \
                   total_bpd_b.shape == prior_bpd_b.shape ==  torch.Size([B])
            return total_bpd_b.mean(), vals_bt_.mean(), prior_bpd_b.mean(), mse_bt_.mean()

def get_target_label(logits, label, device):

    rates, indices = logits.sort(1, descending=True) 
    rates, indices = rates.squeeze(0), indices.squeeze(0)  

    tar_label = torch.zeros_like(label).to(device)

    for i in range(label.shape[0]):
        if label[i] == indices[i][0]:  # classify is correct
            tar_label[i] = indices[i][1]
        else:
            tar_label[i] = indices[i][0]

    return tar_label

class PVCNN2(PVCNN2Base):
    sa_blocks = [
        ((32, 2, 32), (1024, 0.1, 32, (32, 64))),
        ((64, 3, 16), (256, 0.2, 32, (64, 128))),
        ((128, 3, 8), (64, 0.4, 32, (128, 256))),
        (None, (16, 0.8, 32, (256, 256, 512))),
    ]
    fp_blocks = [
        ((256, 256), (256, 3, 8)),
        ((256, 256), (256, 3, 8)),
        ((256, 128), (128, 2, 16)),
        ((128, 128, 64), (64, 2, 32)),
    ]

    def __init__(self, num_classes, sv_points, embed_dim, use_att,dropout, extra_feature_channels=3, width_multiplier=1,
                 voxel_resolution_multiplier=1):
        super().__init__(
            num_classes=num_classes, sv_points=sv_points, embed_dim=embed_dim, use_att=use_att,
            dropout=dropout, extra_feature_channels=extra_feature_channels,
            width_multiplier=width_multiplier, voxel_resolution_multiplier=voxel_resolution_multiplier
        )


class Model(nn.Module):
    def __init__(self, args, betas, loss_type: str, model_mean_type: str, model_var_type:str):
        super(Model, self).__init__()
        self.diffusion = GaussianDiffusion(betas, loss_type, model_mean_type, model_var_type, args.svpoints, args)

        self.model = PVCNN2(num_classes=args.nc, sv_points=args.svpoints, embed_dim=args.embed_dim, use_att=args.attention,
                            dropout=args.drop_out, extra_feature_channels=0)

    def prior_kl(self, x0):
        return self.diffusion._prior_bpd(x0)

    def all_kl(self, x0, clip_denoised=True):
        total_bpd_b, vals_bt, prior_bpd_b, mse_bt =  self.diffusion.calc_bpd_loop(self._denoise, x0, clip_denoised)

        return {
            'total_bpd_b': total_bpd_b,
            'terms_bpd': vals_bt,
            'prior_bpd_b': prior_bpd_b,
            'mse_bt':mse_bt
        }


    def _denoise(self, data, t):
        B, D,N= data.shape
        assert data.dtype == torch.float
        assert t.shape == torch.Size([B]) and t.dtype == torch.int64

        out = self.model(data, t)

        return out

    def get_loss_iter(self, data, noises=None):
        B, D, N = data.shape
        t = torch.randint(0, self.diffusion.num_timesteps, size=(B,), device=data.device)

        if noises is not None:
            noises[t!=0] = torch.randn((t!=0).sum(), *noises.shape[1:]).to(noises)

        losses = self.diffusion.p_losses(
            denoise_fn=self._denoise, data_start=data, t=t, noise=noises)
        assert losses.shape == t.shape == torch.Size([B])
        return losses

    def gen_samples(self, partial_x, shape, device, noise_fn=torch.randn,
                    clip_denoised=True,
                    keep_running=False, adv_model=None, model_index=0):
        return self.diffusion.p_sample_loop(partial_x, self._denoise, shape=shape, device=device, noise_fn=noise_fn,
                                            clip_denoised=clip_denoised,
                                            keep_running=keep_running, adv_models=adv_model, model_index=model_index)


    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def multi_gpu_wrapper(self, f):
        self.model = f(self.model)

def get_betas(schedule_type, b_start, b_end, time_num):
    if schedule_type == 'linear':
        betas = np.linspace(b_start, b_end, time_num)
    elif schedule_type == 'warm0.1':

        betas = b_end * np.ones(time_num, dtype=np.float64)
        warmup_time = int(time_num * 0.1)
        betas[:warmup_time] = np.linspace(b_start, b_end, warmup_time, dtype=np.float64)
    elif schedule_type == 'warm0.2':

        betas = b_end * np.ones(time_num, dtype=np.float64)
        warmup_time = int(time_num * 0.2)
        betas[:warmup_time] = np.linspace(b_start, b_end, warmup_time, dtype=np.float64)
    elif schedule_type == 'warm0.5':

        betas = b_end * np.ones(time_num, dtype=np.float64)
        warmup_time = int(time_num * 0.5)
        betas[:warmup_time] = np.linspace(b_start, b_end, warmup_time, dtype=np.float64)
    else:
        raise NotImplementedError(schedule_type)
    return betas



#############################################################################

def get_mvr_dataset(pc_dataroot, views_root, npoints,category):
    tr_dataset = ShapeNet15kPointClouds(root_dir=pc_dataroot,
        categories=[category], split='train',
        tr_sample_size=npoints,
        te_sample_size=npoints,
        scale=1.,
        normalize_per_shape=False,
        normalize_std_per_axis=False,
        random_subsample=True)
    te_dataset = ShapeNet_Multiview_Points(root_pc=pc_dataroot, root_views=views_root,
                                            cache=os.path.join(pc_dataroot, '../cache'), split='val',
        categories=[category],
        npoints=npoints, sv_samples=200,
        all_points_mean=tr_dataset.all_points_mean,
        all_points_std=tr_dataset.all_points_std,
    )
    return te_dataset


def evaluate_recon_mvr(opt, model, save_dir, logger):
    
    
    num_classes = 16
    
    model1 = PointNetCls(k=55, feature_transform=opt.feature_transform)
    model1 = nn.DataParallel(model1).cuda()
    model1.load_state_dict(torch.load(opt.model1_path))
    model1.eval()
    
    model2 = DGCNN(opt.emb_dims, opt.k, output_channels=55)
    model2 = nn.DataParallel(model2).cuda()
    model2.load_state_dict(torch.load(opt.model2_path))
    model2.eval()
    
    model3 = RPC(opt,output_channels=55)
    model3 = nn.DataParallel(model3).cuda()
    model3.load_state_dict(torch.load(opt.model3_path))
    model3.eval()
    
    adv_models=[model1,model2,model3]
        
        
    defense_module =  SORDefense(k=2, alpha=1.1)
    test_dataset = get_mvr_dataset(opt.dataroot_pc, opt.dataroot_sv,
                                      opt.npoints, opt.category)

    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batch_size,
                                                  shuffle=False, num_workers=int(opt.workers), drop_last=False)
    ref = []
    samples = []
    masked = []
    total_adv = []
    k = 0
    sc_points=1860
    
    for i, data in tqdm(enumerate(test_dataloader), total=len(test_dataloader), desc='Reconstructing Samples'):
            
        adv_samples = []
        adv_masked = []
        adv_index = []
    
        gt_all = data['test_points']
        x_all = data['sv_points']
        x_all = x_all[:,:5,:]
        B,V,N,C = x_all.shape
        gt_all = gt_all[:,None,:,:].expand(-1, V, -1,-1)
        gt_all = gt_all.reshape(-1,N,C)
        
        final_adv = torch.ones(B*V, 3*N - opt.svpoints * (len(adv_models) - 1), C)
        
        x = x_all.reshape(B * V, N, C).transpose(1, 2).contiguous()
        
        #'''
        
        label = torch.tensor(final_adv.shape[0]*[args.category_label]).cuda()


        final_adv[:,:opt.svpoints,:] = x[:,:,:opt.svpoints].transpose(1, 2).contiguous()
        
        m, s = data['mean'].float(), data['std'].float()
        
        recon, index = model.gen_samples(x[:, :, :opt.svpoints].cuda(), x[:, :, opt.svpoints:].shape, 'cuda',
                                      clip_denoised=False, adv_model=adv_models, model_index=0)
        recon = recon.detach().cpu()
        if (recon.shape[0] == 0):
            continue
        BV = recon.shape[0]
        recon = recon.transpose(1, 2).contiguous()
        x = x.transpose(1, 2).contiguous()
        gt_all = gt_all[index].reshape(BV,N,C)
        x_adj = x[index].reshape(BV,N,C)
        recon_adj = recon.reshape(BV,N,C)
        recon_adj = sor(recon_adj, defense_module)
        visualize_pointcloud_batch(os.path.join(str(Path(opt.eval_path).parent), 'x.png'), recon_adj[:64], None,
                           None, None)
        ref.append(gt_all)
        masked.append(x_adj[:,:test_dataloader.dataset.sv_samples,:])
        samples.append(recon_adj)

    ref_pcs = torch.cat(ref, dim=0)
    sample_pcs = torch.cat(samples, dim=0)
    masked = torch.cat(masked, dim=0)
    BV, N, C = ref_pcs.shape

    np.savez(os.path.join(save_dir, '1bl_adv.pth'),
     test_pc=sample_pcs.cpu().numpy().astype(np.float32),
     test_label=np.array(sample_pcs.shape[0]*[args.category_label]).astype(np.uint8),
     target_label=np.array(sample_pcs.shape[0]*[args.category_label]).astype(np.uint8),
     ori_pc=ref_pcs.cpu().numpy().astype(np.float32))


    results = EMD_CD(sample_pcs.reshape(BV, N, C),
                     ref_pcs.reshape(BV, N, C), opt.batch_size, reduced=False)

    torch.save(ref_pcs.reshape(BV, N, C), os.path.join(save_dir, 'recon_gt.pth'))

    torch.save(masked.reshape(BV, *masked.shape[2:]), os.path.join(save_dir, 'recon_masked.pth'))
    # Compute metrics

    results = {ky: val.reshape(B,V) if val.shape == torch.Size([B*V,]) else val for ky, val in results.items()}

    pprint({key: val.mean().item() for key, val in results.items()})
    logger.info({key: val.mean().item() for key, val in results.items()})

    results['pc'] = sample_pcs
    torch.save(results, os.path.join(save_dir, 'ours_results.pth'))
        
    del ref_pcs, masked, results

def evaluate_saved(opt, saved_dir):
    # ours_base = '/viscam/u/alexzhou907/research/diffusion/shape_completion/output/test_chair/2020-11-04-02-10-38/syn'

    gt_pth = saved_dir + '/recon_gt.pth'
    ours_pth = saved_dir + '/ours_results.pth'
    gt = torch.load(gt_pth).permute(1,0,2,3)
    ours = torch.load(ours_pth)['pc'].permute(1,0,2,3)

    all_res = {}
    for i, (gt_, ours_) in enumerate(zip(gt, ours)):
        results = compute_all_metrics(gt_, ours_, opt.batch_size)

        for key, val in results.items():
            if i == 0:
                all_res[key] = val
            else:
                all_res[key] += val
        pprint(results)
    for key, val in all_res.items():
        all_res[key] = val / gt.shape[0]

    pprint({key: val.mean().item() for key, val in all_res.items()})



def main(opt):
    exp_id = os.path.splitext(os.path.basename(__file__))[0]
    dir_id = os.path.dirname(__file__)
    output_dir = get_output_dir(dir_id, exp_id)
    copy_source(__file__, output_dir)
    logger = setup_logging(output_dir)

    outf_syn, = setup_output_subdirs(output_dir, 'syn')

    betas = get_betas(opt.schedule_type, opt.beta_start, opt.beta_end, opt.time_num)
    model = Model(opt, betas, opt.loss_type, opt.model_mean_type, opt.model_var_type)

    if opt.cuda:
        model.cuda()

    def _transform_(m):
        return nn.parallel.DataParallel(m)

    model = model.cuda()
    model.multi_gpu_wrapper(_transform_)

    model.eval()
    
    opt.eval_path = os.path.join(outf_syn, 'samples.pth')
    Path(opt.eval_path).parent.mkdir(parents=True, exist_ok=True)
    
    with torch.no_grad():

        logger.info("Resume Path:%s" % opt.model)

        resumed_param = torch.load(opt.model)
        model.load_state_dict(resumed_param['model_state'])


        if opt.eval_recon_mvr:
            # Evaluate generation
            evaluate_recon_mvr(opt, model, outf_syn, logger)

        if opt.eval_saved:
            evaluate_saved(opt, outf_syn)


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot_pc', default='/ShapeNetCore.v2.PC15k/')
    parser.add_argument('--dataroot_sv', default='/ShapeNetC/')
    parser.add_argument('--category', default='chair')
    parser.add_argument('--category_label', type=int, default=14, help='category label from the target model')

    parser.add_argument('--batch_size', type=int, default=5, help='input batch size')
    parser.add_argument('--workers', type=int, default=16, help='workers')
    parser.add_argument('--niter', type=int, default=10000, help='number of epochs to train for')

    parser.add_argument('--eval_recon_mvr', default=True)
    parser.add_argument('--eval_saved', default=True)

    parser.add_argument('--nc', default=3)
    parser.add_argument('--npoints', default=2060)
    parser.add_argument('--svpoints', default=200)
    '''model'''
    parser.add_argument('--beta_start', default=0.0001)
    parser.add_argument('--beta_end', default=0.02)
    parser.add_argument('--schedule_type', default='linear')
    parser.add_argument('--time_num', default=1000)

    #params
    parser.add_argument('--attention', default=True)
    parser.add_argument('--drop_out', default=0.1)
    parser.add_argument('--embed_dim', type=int, default=64)
    parser.add_argument('--loss_type', default='mse')
    parser.add_argument('--model_mean_type', default='eps')
    parser.add_argument('--model_var_type', default='fixedsmall')
    
    parser.add_argument('--num_points', type=int, default=2048,
                        help='num of points to use')
    parser.add_argument('--emb_dims', type=int, default=2048, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--feature_transform', type=str2bool, default=True,
                        help='whether to use STN on features in PointNet')
    parser.add_argument('--k', type=int, default=20, metavar='N',
                        help='Num of nearest neighbors to use')
    parser.add_argument('--model1_path', type=str, default='/pretrain/modelnet/pointnet.pth',
                        help='Model weight to load, use config if not specified')
    parser.add_argument('--model2_path', type=str, default='/pretrain/modelnet/dgcnn.pth',
                        help='Model weight to load, use config if not specified')
    parser.add_argument('--model3_path', type=str, default='/pretrain/modelnet/rpc.pth',
                        help='Model weight to load, use config if not specified')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
        

    parser.add_argument('--model', default='checkpoint/completion/chair_completion_2799.pth', help="path to model (to continue training)")

    '''eval'''

    parser.add_argument('--eval_path',
                        default='eval/')

    parser.add_argument('--manualSeed', default=42, type=int, help='random seed')

    parser.add_argument('--gpu', type=int, default=0, metavar='S', help='gpu id (default: 0)')

    opt = parser.parse_args()

    if torch.cuda.is_available():
        opt.cuda = True
    else:
        opt.cuda = False

    return opt
if __name__ == '__main__':
    opt = parse_args()

    main(opt)
