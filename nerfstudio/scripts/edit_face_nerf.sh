# Please change {timestamp} to the timestamp of the model you want to use.

ns-train dc --data ../dataset/face/ --load-dir ./outputs/face/nerfacto/{timestamp}/nerfstudio_models/ \
 --pipeline.dc.src_prompt "a photo of a face" \
 --pipeline.dc.tgt_prompt "Turn him into the Tolkien Elf" \
 --vis viewer \
 --pipeline.dc_freeu_b1 1.1 \
 --pipeline.dc_freeu_b2 1.1 \
 --pipeline.dc_freeu_s1 0.9 \
 --pipeline.dc_freeu_s2 0.2 \
 --pipeline.dc.sd_pretrained_model_or_path timbrooks/instruct-pix2pix \
 --viewer.websocket-port 7007;

ns-train dc --data ../dataset/face/ --load-dir ./outputs/face/nerfacto/{timestamp}/nerfstudio_models/ \
 --pipeline.dc.src_prompt "a photo of a face" \
 --pipeline.dc.tgt_prompt "Turn him into Emma Watson" \
 --vis viewer \
 --pipeline.dc_freeu_b1 1.1 \
 --pipeline.dc_freeu_b2 1.1 \
 --pipeline.dc_freeu_s1 0.9 \
 --pipeline.dc_freeu_s2 0.2 \
 --pipeline.dc.sd_pretrained_model_or_path timbrooks/instruct-pix2pix \
 --viewer.websocket-port 7007;

ns-train dc --data ../dataset/face/ --load-dir ./outputs/face/nerfacto/{timestamp}/nerfstudio_models/ \
 --pipeline.dc.src_prompt "a photo of a face" \
 --pipeline.dc.tgt_prompt "Turn him into an Einstein" \
 --vis viewer \
 --pipeline.dc_freeu_b1 1.1 \
 --pipeline.dc_freeu_b2 1.1 \
 --pipeline.dc_freeu_s1 0.9 \
 --pipeline.dc_freeu_s2 0.2 \
 --pipeline.dc.sd_pretrained_model_or_path timbrooks/instruct-pix2pix \
 --viewer.websocket-port 7007; 