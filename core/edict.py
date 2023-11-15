import torch
import PIL.Image as Image

from diffusers import StableDiffusionPipeline
from diffusers.models.attention_processor import Attention


def hasnan(t):
    return torch.any(torch.isnan(t))

class EDICT(StableDiffusionPipeline):

    def decode_latents(self, latents):

        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents, return_dict=False)[0]
        # image = (image / 2 + 0.5).clamp(0, 1)

        return image

    def encode_string(
        self,
        prompt=None,
        prompt_embeds=None,
    ):
        if prompt_embeds is not None:
            return prompt_embeds

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(self._execution_device, dtype=torch.long)

        if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
            attention_mask = text_inputs.attention_mask.to(self._execution_device)
        else:
            attention_mask = None

        prompt_embeds = self.text_encoder(
            text_input_ids.to(self._execution_device),
            attention_mask=attention_mask,
        )[0].to(self._execution_device, dtype=self.text_encoder.dtype)

        return prompt_embeds

    def encode_prompt(
        self,
        prompt,
        prompt_embeds,
        do_classifier_free_guidance,
    ):
        assert (prompt is not None) != (prompt_embeds is not None), 'only "prompt" or "prompt_embeds" can be None, not both or neither'

        if prompt_embeds is None:
            prompt_embeds = self.encode_string(prompt=prompt)

        uncond = None
        if do_classifier_free_guidance:
            uncond = self.encode_string(prompt='')
            uncond = uncond.expand(prompt_embeds.size(0), -1, -1)

        return prompt_embeds, uncond

    @staticmethod
    def torch_to_pil(img):
        img = ((img + 1) / 2).permute(0, 2, 3, 1).detach().cpu()
        img = img.clamp(0, 1).numpy()
        return Image.fromarray(img)

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

        # EDICT inputs
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

        yt = xt.detach()

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
            
            # Stabilization
            yt = (yt - (1 - p) * xt) / p
            xt = (xt - (1 - p) * yt) / p

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

            yt = self.reverse_scheduler.step(eps, t, yt, eta=eta).prev_sample

            eps = self.unet(
                yt,
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

        return xt, yt, x0_dec, xt_dec, feats  # Return cond and uncond embeddings just in case

    def Denoise(
        self,
        prompt,
        prompt_embeds,
        xt,
        yt,
        num_inference_steps,
        total_num_inference_steps,
        guidance_scale=7.5,
        eta=0.0,
        generator=None,
        cross_attention_kwargs=None,

        decode=True,
        return_pil=True,

        # EDICT inputs
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

            # update l2 loss
            if l2 > 0.0:
                # xt = l2 * xt + (1 - l2) * feats[t.item()][0]
                # yt = l2 * yt + (1 - l2) * feats[t.item()][1]
                xt = xt - l2 * (xt - feats[t.item()][0].to(self.device))
                yt = yt - l2 * (yt - feats[t.item()][1].to(self.device))

            eps = self.unet(
                yt,
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

            yt = self.scheduler.step(eps, t, yt, eta=eta).prev_sample

            # Stabilization
            xt = p * xt + (1 - p) * yt
            yt = p * yt + (1 - p) * xt

        decoded = self.decode_latents(xt)
        if return_pil:
            decoded = self.numpy_to_pil(decoded)

        return decoded

