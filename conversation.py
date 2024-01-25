from transformers import AutoProcessor, BarkModel
from scipy.io.wavfile import write as write_wav
from model import Conversation, Message, MessageBlock
import torch
import numpy as np
import gc
import json
import aiohttp
from dotenv import load_dotenv, dotenv_values

load_dotenv()
env_vars = dotenv_values(".env")

async def make_async_post_request(url, data, headers):
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=data, headers=headers) as response:
            return await response.text(encoding='utf-8')

async def bark_message(message: Message, processor: AutoProcessor, model: BarkModel, speaker: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = processor(message, voice_preset=speaker)
    inputs.to(device)
    
    audio_array = model.generate(
        **inputs,        
        pad_token_id=processor.tokenizer.pad_token_id
    )
    
    audio_array = audio_array.cpu().numpy().squeeze()
    
    return audio_array

async def translate_message(message: Message, language_from: str, language_to: str):
    url = "https://swift-translate.p.rapidapi.com/translate"

    payload = {
        "text": message,
        "sourceLang": language_from,
        "targetLang": language_to
    }
    headers = {
        "content-type": "application/json",
        "X-RapidAPI-Key": env_vars["RAPIDAPI_KEY"],
        "X-RapidAPI-Host": "swift-translate.p.rapidapi.com"
    }
    
    response = await make_async_post_request(url, data=payload, headers=headers)
    response = json.loads(response)
    if response['translatedText']:        
        return response['translatedText']
    else:
        return ""    

# Go forth and be free
async def free_cuda():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    gc.collect()


# TODO: This behaves strangely, even if the alternating method is much slower,
# it is complicated to gget good results out of this.
# The idea is to process ALL messages for one speaker in batch, and with those audios
# reconstruct the conversation.
async def bark_batch_test(conversation_model: Conversation):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    try: 
        processor = AutoProcessor.from_pretrained(conversation_model.bark_model, device_map = device)
        model = BarkModel.from_pretrained(conversation_model.bark_model, torch_dtype=torch.float16).to(device)

        # This comes from here: https://huggingface.co/blog/optimizing-bark
        text_prompt = [
            "Let's try generating speech, with Bark, a text-to-speech model",
            "Wow, batching is so great!",
            "I love Hugging Face, it's so cool."
        ]

        inputs = processor(text_prompt).to(device)

        with torch.inference_mode():
            audio_array = model.generate(**inputs, do_sample = True, fine_temperature = 0.4, coarse_temperature = 0.8)
        
        sample_rate = model.generation_config.sample_rate

        for i, audio in enumerate(audio_array):
            audio = audio.cpu().numpy().squeeze()
            if np.any(audio):
                audio = (audio * np.iinfo(np.int32).max).astype(np.int32)
                # These wavs are very wonky, not really usable
                write_wav(f"bark_generation_{i}.wav", sample_rate, audio)

        await free_cuda()
        
    except Exception as e:
        print(e)


# TODO: Process every speaker in batch instead of alternating between them
# TODO: Implement another translation method and process in batch as well
async def bark_conversation(conversation_model: Conversation):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    try: 
        processor = AutoProcessor.from_pretrained(conversation_model.bark_model, device_map = device)
        model = BarkModel.from_pretrained(conversation_model.bark_model, torch_dtype=torch.float16).to(device)

        # Make it go faster.
        model.enable_cpu_offload()        
        model = model.to_bettertransformer()

        combined_audio_list = []
        sample_rate = model.generation_config.sample_rate
        
        delay_sec = 0.12 #THIS VALUE IS LANGUAGE-DEPENDENT FOR SURE. BUT I'M SPANISH, SO...
        delay_samples = int(delay_sec * sample_rate)
        silence = np.zeros(delay_samples, dtype=np.int32)
        
        for message in conversation_model.messages:
            if hasattr(conversation_model, 'translateFromTo') and conversation_model.translateFromTo != None:                
                message.message = await translate_message(message.message, conversation_model.translateFromTo[0], conversation_model.translateFromTo[1])
                speaker_model = conversation_model.translate_speakers_mapping.get(message.speaker)
            else:                
                speaker_model = conversation_model.speaker_mapping.get(message.speaker)
            
            audio_element = await bark_message(message.message, processor, model, speaker_model)
            
            if np.any(audio_element):
                # Conversion of audio for "lossless" concatenation, sometimes throws warning, but works OK, I think.                
                audio_element = (audio_element * np.iinfo(np.int32).max).astype(np.int32)
                combined_audio_list.append(audio_element)
                combined_audio_list.append(silence)
            else:
                print(f"Warning: Audio element for {message.speaker} is silent.")
            
        combined_audio = np.concatenate(combined_audio_list)
        await free_cuda()
        return combined_audio, sample_rate
    except Exception as e:
        print(e)
    