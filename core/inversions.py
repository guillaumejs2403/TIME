import torch
import PIL.Image as Image

from diffusers import StableDiffusionPipeline
from diffusers.models.attention_processor import Attention

from .edict import EDICT

def hasnan(t):
    return torch.any(torch.isnan(t))

class DDIMInversion(EDICT):

    def Invert(
        self,
        prompt,
        prompt_embeds,
        image,
        num_inference_steps,
        total_num_inference_steps,
        guidance_scale=7.5,
        eta=0.0,
        generator=None,
        cross_attention_kwargs=None,

        decode=True,
        return_pil=True,

        # EDICT inputs, useless rn
        p=0.93,
        negative_prompt=None,
        negative_prompt_embeds=None,
        l2=0.0,
    ):
        assert hasattr(self, 'reverse_scheduler'), 'Pipeline must has "reverse_scheduler" scheduler. Initialize it with pipeline.reverse_scheduler = DDIMInverseScheduler'

        do_classifier_free_guidance = guidance_scale != 1.0
        do_negative_guidance = (negative_prompt is not None) or (negative_prompt_embeds is not None)

        # Prepare variables
        device = self._execution_device
        # extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
        extra_step_kwargs = {'eta': eta}

        # Prepare timesteps
        self.reverse_scheduler.set_timesteps(total_num_inference_steps, device=device)
        timesteps = self.reverse_scheduler.timesteps

        # Create Latents
        xt = self.vae.encode(image).latent_dist.mean
        xt = self.vae.config.scaling_factor * xt

        x0_dec = None
        if decode:
            x0_dec = self.decode_latents(xt.detach())
            if return_pil:
                x0_dec = self.torch_to_pil(x0_dec)

        # Create prompt embeddings if not given
        cond, uncond = self.encode_prompt(
            prompt=prompt,
            prompt_embeds=prompt_embeds,
            do_classifier_free_guidance=do_classifier_free_guidance and not do_negative_guidance,
        )

        if do_negative_guidance and do_classifier_free_guidance:
            uncond, _ = self.encode_prompt(
                prompt=negative_prompt,
                prompt_embeds=negative_prompt_embeds,
                do_classifier_free_guidance=False,
            )

        feats = {} if (l2 > 0) else None

        for t in timesteps[:num_inference_steps]:  # still need to set this to something intermediate

            eps = self.unet(
                xt,
                t,
                encoder_hidden_states=cond,
                cross_attention_kwargs=cross_attention_kwargs
            ).sample

            if do_classifier_free_guidance:
                eps_uncond = self.unet(
                    xt,
                    t,
                    encoder_hidden_states=uncond,
                    cross_attention_kwargs=cross_attention_kwargs
                ).sample

                eps = eps_uncond + guidance_scale * (eps - eps_uncond)

            xt = self.reverse_scheduler.step(eps, t, xt, eta=eta).prev_sample

            if torch.any(torch.isnan(xt)) or torch.any(torch.isnan(yt)):
                print('Found nan!')
                import ipdb; ipdb.set_trace()

            if l2 > 0:
                feats[t.item()] = [xt.detach().cpu(), yt.detach().cpu()]

        xt_dec = None
        if decode:
            xt_dec = self.decode_latents(xt.detach())
            if return_pil:
                xt_dec = self.numpy_to_pil(xt_dec)

        return xt, xt, x0_dec, xt_dec, feats  # Return cond and uncond embeddings just in case

    def Denoise(
        self,
        prompt,
        prompt_embeds,
        xt,
        yt,  # useless
        num_inference_steps,
        total_num_inference_steps,
        guidance_scale=7.5,
        eta=0.0,
        generator=None,
        cross_attention_kwargs=None,

        decode=True,
        return_pil=True,

        # EDICT inputs, useless
        p=0.93,
        negative_prompt=None,
        negative_prompt_embeds=None,
        l2=0.0,
        feats=None
    ):
        assert hasattr(self, 'reverse_scheduler'), 'Pipeline must has "reverse_scheduler" scheduler. Initialize it with pipeline.reverse_scheduler = DDIMInverseScheduler'
        assert not ((feats is None) and (l2 > 0)), 'If l2 is higher than 0, then feats should be differennt than None'

        do_classifier_free_guidance = guidance_scale != 1.0
        do_negative_guidance = negative_prompt is not None or negative_prompt_embeds is not None

        # Prepare variables
        device = self._execution_device
        # extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
        extra_step_kwargs = {'eta': eta}

        # Prepare timesteps
        self.scheduler.set_timesteps(total_num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # Create prompt embeddings if not given
        cond, uncond = self.encode_prompt(
            prompt=prompt,
            prompt_embeds=prompt_embeds,
            do_classifier_free_guidance=do_classifier_free_guidance and not do_negative_guidance,
        )

        if do_negative_guidance and do_classifier_free_guidance:
            uncond, _ = self.encode_prompt(
                prompt=negative_prompt,
                prompt_embeds=negative_prompt_embeds,
                do_classifier_free_guidance=False,
            )

        for t in timesteps[-num_inference_steps:]:

            eps = self.unet(
                xt,
                t,
                encoder_hidden_states=cond,
                cross_attention_kwargs=cross_attention_kwargs
            ).sample

            if do_classifier_free_guidance:
                eps_uncond = self.unet(
                    yt,
                    t,
                    encoder_hidden_states=uncond,
                    cross_attention_kwargs=cross_attention_kwargs
                ).sample
                eps = eps_uncond + guidance_scale * (eps - eps_uncond)
            

            xt = self.scheduler.step(eps, t, xt, eta=eta).prev_sample

        decoded = self.decode_latents(xt)
        if return_pil:
            decoded = self.numpy_to_pil(decoded)

        return decoded


class NaiveInversion(EDICT):

    def Invert(
        self,
        prompt,
        prompt_embeds,
        image,
        num_inference_steps,
        total_num_inference_steps,
        guidance_scale=7.5,
        eta=0.0,
        generator=None,
        cross_attention_kwargs=None,

        decode=True,
        return_pil=True,

        # EDICT inputs, useless rn
        p=0.93,
        negative_prompt=None,
        negative_prompt_embeds=None,
        l2=0.0,
    ):
        assert hasattr(self, 'reverse_scheduler'), 'Pipeline must has "reverse_scheduler" scheduler. Initialize it with pipeline.reverse_scheduler = DDIMInverseScheduler'

        do_classifier_free_guidance = guidance_scale != 1.0
        do_negative_guidance = (negative_prompt is not None) or (negative_prompt_embeds is not None)

        # Prepare variables
        device = self._execution_device
        # extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
        extra_step_kwargs = {'eta': eta}

        # Prepare timesteps
        self.reverse_scheduler.set_timesteps(total_num_inference_steps, device=device)
        timesteps = self.reverse_scheduler.timesteps

        # Create Latents
        xt = self.vae.encode(image).latent_dist.mean
        xt = self.vae.config.scaling_factor * xt

        x0_dec = None
        if decode:
            x0_dec = self.decode_latents(xt.detach())
            if return_pil:
                x0_dec = self.torch_to_pil(x0_dec)

        # Create prompt embeddings if not given
        cond, uncond = self.encode_prompt(
            prompt=prompt,
            prompt_embeds=prompt_embeds,
            do_classifier_free_guidance=do_classifier_free_guidance and not do_negative_guidance,
        )

        if do_negative_guidance and do_classifier_free_guidance:
            uncond, _ = self.encode_prompt(
                prompt=negative_prompt,
                prompt_embeds=negative_prompt_embeds,
                do_classifier_free_guidance=False,
            )

        feats = {} if (l2 > 0) else None

        xt = self.scheduler.add_noise(
            original_samples=xt,
            noise=torch.randn_like(xt),
            timesteps=torch.ones(xt.size(0),
                                 dtype=torch.long,
                                 device=device) * timesteps[num_inference_steps]
        )

        xt_dec = None
        if decode:
            xt_dec = self.decode_latents(xt.detach())
            if return_pil:
                xt_dec = self.numpy_to_pil(xt_dec)

        return xt, xt, x0_dec, xt_dec, feats  # Return cond and uncond embeddings just in case

    def Denoise(
        self,
        prompt,
        prompt_embeds,
        xt,
        yt,  # useless
        num_inference_steps,
        total_num_inference_steps,
        guidance_scale=7.5,
        eta=0.0,
        generator=None,
        cross_attention_kwargs=None,

        decode=True,
        return_pil=True,

        # EDICT inputs, useless
        p=0.93,
        negative_prompt=None,
        negative_prompt_embeds=None,
        l2=0.0,
        feats=None
    ):
        assert hasattr(self, 'reverse_scheduler'), 'Pipeline must has "reverse_scheduler" scheduler. Initialize it with pipeline.reverse_scheduler = DDIMInverseScheduler'
        assert not ((feats is None) and (l2 > 0)), 'If l2 is higher than 0, then feats should be differennt than None'

        do_classifier_free_guidance = guidance_scale != 1.0
        do_negative_guidance = negative_prompt is not None or negative_prompt_embeds is not None

        # Prepare variables
        device = self._execution_device
        # extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
        extra_step_kwargs = {'eta': eta}

        # Prepare timesteps
        self.scheduler.set_timesteps(total_num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # Create prompt embeddings if not given
        cond, uncond = self.encode_prompt(
            prompt=prompt,
            prompt_embeds=prompt_embeds,
            do_classifier_free_guidance=do_classifier_free_guidance and not do_negative_guidance,
        )

        if do_negative_guidance and do_classifier_free_guidance:
            uncond, _ = self.encode_prompt(
                prompt=negative_prompt,
                prompt_embeds=negative_prompt_embeds,
                do_classifier_free_guidance=False,
            )

        for t in timesteps[-num_inference_steps:]:

            eps = self.unet(
                xt,
                t,
                encoder_hidden_states=cond,
                cross_attention_kwargs=cross_attention_kwargs
            ).sample

            if do_classifier_free_guidance:
                eps_uncond = self.unet(
                    yt,
                    t,
                    encoder_hidden_states=uncond,
                    cross_attention_kwargs=cross_attention_kwargs
                ).sample
                eps = eps_uncond + guidance_scale * (eps - eps_uncond)
            

            xt = self.scheduler.step(eps, t, xt, eta=eta).prev_sample

        decoded = self.decode_latents(xt)
        if return_pil:
            decoded = self.numpy_to_pil(decoded)

        return decoded
