from fastapi import FastAPI, HTTPException, Request
from conversation import bark_conversation, bark_batch_test
from model import Conversation, Message
from fastapi.responses import StreamingResponse
from io import BytesIO
import scipy

app = FastAPI()

@app.post("/conversation")
async def process_data(request: Request):
    try:
        data = await request.json()
        conversation = Conversation.create_with_mapping(**data)
        # testing batches here, NOT WORKING YET
        # await bark_batch_test(conversation)
        audio_sample, sample_rate = await bark_conversation(conversation)
        audio_bytes_io = BytesIO()        
        scipy.io.wavfile.write(audio_bytes_io, rate=sample_rate, data=audio_sample)
        content_type = "audio/wav"
        return StreamingResponse(audio_bytes_io, media_type=content_type)
    except Exception as e:
        print(e)
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn    
    uvicorn.run(app, host="127.0.0.1", port=8000)
