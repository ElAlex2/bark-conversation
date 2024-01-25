from typing import List, Tuple, Optional
from pydantic import BaseModel, Field
import json

class Message(BaseModel):
    speaker: str
    message: str

class MessageBlock(BaseModel):
    speaker: str
    messages: List[str]
    message_for_translation: str

class Conversation(BaseModel):
    speakers: List[Tuple[str, str]]
    speaker_mapping: dict = Field(default_factory=dict)
    messages: List[Message]
    messages_block_mapped: Optional[List[MessageBlock]] = None    
    bark_model: Optional[str] = "suno/bark-small"
    translateFromTo: Optional[Tuple[str, str]] = None
    translateToSpeakers: Optional[List[Tuple[str, str]]] = None
    translate_speakers_mapping: Optional[dict] = Field(default_factory=dict)

    @classmethod
    def create_with_mapping(cls, speakers: List[Tuple[str, str]], messages: List[Message], **kwargs):        
        speaker_mapping = dict(speakers)
        messages_block_mapped = []

        for message_data in messages:
            speaker = message_data["speaker"]
            message = message_data["message"]
            
            existing_block = next((block for block in messages_block_mapped if block.speaker == speaker), None)

            if existing_block:                
                existing_block.messages.append(message)
                existing_block.message_for_translation = existing_block.message_for_translation + "|" + message
            else:                
                new_block = MessageBlock(speaker=speaker, messages=[message], message_for_translation=message)
                messages_block_mapped.append(new_block)

        if 'translateToSpeakers' in kwargs and 'translateFromTo' in kwargs:
            translate_speakers_mapping = dict(kwargs['translateToSpeakers'])            
        else:            
            translate_speakers_mapping = None
        
        return cls(speakers=speakers, speaker_mapping=speaker_mapping, messages=messages, 
                   translate_speakers_mapping=translate_speakers_mapping, messages_block_mapped=messages_block_mapped, **kwargs)
