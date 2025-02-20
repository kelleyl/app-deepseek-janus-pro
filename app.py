import argparse
import logging
import yaml
from pathlib import Path
import tqdm
import time
from PIL import Image

from clams import ClamsApp, Restifier
from clams.appmetadata import AppMetadata
from mmif import Mmif, View, Document, AnnotationTypes, DocumentTypes
from mmif.utils import video_document_helper as vdh
from transformers import AutoModelForCausalLM
from janus.models import VLChatProcessor
import torch


class JanusProCaptioner(ClamsApp):

    def __init__(self):
        model_path = "deepseek-ai/Janus-Pro-7B"
        self.vl_chat_processor = VLChatProcessor.from_pretrained(model_path)
        self.tokenizer = self.vl_chat_processor.tokenizer
        self.vl_gpt = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
        
        # Check if GPU supports bfloat16
        if torch.cuda.is_available() and torch.cuda.get_device_capability(0) >= (7, 0):
            self.vl_gpt = self.vl_gpt.to(torch.bfloat16).cuda().eval()
        else:
            self.vl_gpt = self.vl_gpt.to(torch.float16).cuda().eval()
        
        super().__init__()

    def _appmetadata(self) -> AppMetadata:
        # Update metadata for JanusProCaptioner as needed
        pass
    
    def load_config(self, config_file):
        with open(config_file, 'r') as f:
            return yaml.safe_load(f)

    def get_prompt(self, label: str, config: dict) -> str:
        if 'custom_prompts' in config and label in config['custom_prompts']:
            return config['custom_prompts'][label]
        return config['default_prompt']

    def _annotate(self, mmif: Mmif, **parameters) -> Mmif:
        self.logger.debug(f"Annotating with parameters: {parameters}")
        config_file = parameters.get('config')
        print("config_file: ", config_file)
        config_dir = Path(__file__).parent
        config_file = config_dir / config_file
        config = self.load_config(config_file)
        
        batch_size = 8
        new_view: View = mmif.new_view()
        self.sign_view(new_view, parameters)
        new_view.new_contain(DocumentTypes.TextDocument)
        new_view.new_contain(AnnotationTypes.Alignment)

        def process_batch(prompts_batch, images_batch, annotations_batch):
            try:
                all_conversations = []
                for prompt, image in zip(prompts_batch, images_batch):
                    conversation = [
                        {"role": "<|User|>", "content": f"<image_placeholder>\n{prompt}", "images": [image]},
                        {"role": "<|Assistant|>", "content": ""}
                    ]
                    all_conversations.append(conversation)

                # Use the already loaded PIL images directly
                all_pil_images = images_batch

                prepare_inputs = self.vl_chat_processor(conversations=all_conversations, images=all_pil_images, force_batchify=True).to(self.vl_gpt.device)
                inputs_embeds = self.vl_gpt.prepare_inputs_embeds(**prepare_inputs)
                outputs = self.vl_gpt.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=prepare_inputs.attention_mask,
                    pad_token_id=self.tokenizer.eos_token_id,
                    bos_token_id=self.tokenizer.bos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    max_new_tokens=512,
                    do_sample=False,
                    use_cache=True,
                )
                generated_texts = [self.tokenizer.decode(seq.cpu().tolist(), skip_special_tokens=True) for seq in outputs]

                for generated_text, annotation in zip(generated_texts, annotations_batch):
                    text_document = new_view.new_textdocument(generated_text.strip())
                    alignment = new_view.new_annotation(AnnotationTypes.Alignment)
                    alignment.add_property("source", annotation['source'])
                    alignment.add_property("target", text_document.long_id)
            except Exception as e:
                self.logger.error(f"Error processing batch: {e}")
                raise
            finally:
                torch.cuda.empty_cache()

        input_context = config['context_config']['input_context']
        
        if input_context == "image":
            print("input_context: image")
            image_docs = mmif.get_documents_by_type(DocumentTypes.ImageDocument)
            
            for i in range(0, len(image_docs), batch_size):
                batch_docs = image_docs[i:i + batch_size]
                prompts = [config['default_prompt']] * len(batch_docs)
                images = [Image.open(doc.location_path()) for doc in batch_docs]
                annotations_batch = [{'source': doc.long_id} for doc in batch_docs]
                
                start_time = time.time()
                process_batch(prompts, images, annotations_batch)
                print(f"Processed batch of {len(batch_docs)} in {time.time() - start_time:.2f} seconds")
            
        elif input_context == 'timeframe':
            print("input_context: ", input_context)
            app_uri = config['context_config']['timeframe']['app_uri']
            all_views = mmif.get_all_views_contain(AnnotationTypes.TimeFrame)
            timeframes = []
            for view in all_views:
                print(view.metadata.app)
                if app_uri in view.metadata.app:
                    print("found view with app_uri: ", app_uri)
                    timeframes = view.get_annotations(AnnotationTypes.TimeFrame)
                    break
            if not timeframes:
                raise ValueError(f"No TimeFrame annotations found for app_uri: {app_uri}")
            label_mapping = config['context_config']['timeframe'].get('label_mapping', {})
        elif input_context == 'fixed_window':
            print("input_context: ", input_context)
            video_docs = mmif.get_documents_by_type(DocumentTypes.VideoDocument)
            if not video_docs:
                raise ValueError("No VideoDocument found in MMIF.")
            video_doc = video_docs[0]
            window_duration = config['context_config']['fixed_window']['window_duration']
            stride = config['context_config']['fixed_window']['stride']
            fps = float(video_doc.get_property('fps'))
            total_frames = int(video_doc.get_property('frameCount'))
            frame_numbers = list(range(0, total_frames, int(fps * stride)))
        else:
            raise ValueError(f"Unsupported input context: {input_context}")

        if input_context == 'timeframe':
            timeframes = list(timeframes)
            frame_numbers = [vdh.get_mid_framenum(mmif, timeframe) for timeframe in timeframes]
            all_images = vdh.extract_frames_as_images(video_doc, frame_numbers, as_PIL=True)
            
            for i in tqdm.tqdm(range(0, len(timeframes), batch_size)):
                batch_timeframes = timeframes[i:i + batch_size]
                batch_images = all_images[i:i + batch_size]
                
                prompts = []
                annotations_batch = []
                for timeframe in batch_timeframes:
                    label = timeframe.get_property('label')
                    mapped_label = label_mapping.get(label, 'default')
                    prompt = self.get_prompt(mapped_label, config)
                    prompts.append(prompt)
                    annotations_batch.append({'source': timeframe.long_id})
                
                start_time = time.time()
                process_batch(prompts, batch_images, annotations_batch)
                print(f"Processed batch of {len(batch_timeframes)} in {time.time() - start_time:.2f} seconds")

        elif input_context == 'fixed_window':
            prompts = []
            images_batch = []
            annotations_batch = []
            for frame_number in tqdm.tqdm(frame_numbers):
                image = vdh.extract_frame_as_image(video_doc, frame_number, as_PIL=True)
                prompt = config['default_prompt']
                prompts.append(prompt)
                images_batch.append(image)
                timepoint = new_view.new_annotation(AnnotationTypes.TimePoint)
                timepoint.add_property("timePoint", frame_number)
                annotations_batch.append({'source': timepoint.long_id})

                if len(prompts) == batch_size:
                    start_time = time.time()
                    process_batch(prompts, images_batch, annotations_batch)
                    end_time = time.time()
                    elapsed_time = end_time - start_time
                    print(f"Processed a batch of {batch_size} in {elapsed_time:.2f} seconds.")
                    prompts, images_batch, annotations_batch = [], [], []

            if prompts:
                start_time = time.time()
                process_batch(prompts, images_batch, annotations_batch)
                end_time = time.time()
                elapsed_time = end_time - start_time
                print(f"Processed the final batch of {len(prompts)} in {elapsed_time:.2f} seconds.")

        return mmif

    

def get_app():
    return JanusProCaptioner()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", action="store", default="5000", help="set port to listen")
    parser.add_argument("--production", action="store_true", help="run gunicorn server")

    parsed_args = parser.parse_args()

    app = JanusProCaptioner()

    http_app = Restifier(app, port=int(parsed_args.port))

    if parsed_args.production:
        http_app.serve_production()
    else:
        app.logger.setLevel(logging.DEBUG)
        http_app.run() 