from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
import ctranslate2
import transformers
import time

# init fastapi
app = FastAPI()

# preload model on startup
@app.on_event("startup")
def load_model():
    global tokenizer
    global translator
    # load tokenizer and model
    translator = ctranslate2.Translator("nq2sq-ct", compute_type="int8")
    tokenizer = transformers.AutoTokenizer.from_pretrained("rehanzo/nq2sq")

class Question(BaseModel):
    question: str

# api runs on root
# we end up having its 'root' be '/api/' for the live demo
@app.post("/")
async def query_gen(question_data: Question):
    input_text = question_data.question
    
    # tokenize
    input_tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(input_text))

    # params tweaked to optimize output
    # 1.3 repetition penalty seems most optimal to balance response quality and prevention of repeating entire queries over and over
    results = translator.translate_batch([input_tokens], repetition_penalty=1.3, sampling_temperature=1, beam_size=2)

    output_tokens = results[0].hypotheses[0]
    queries = tokenizer.decode(tokenizer.convert_tokens_to_ids(output_tokens))

    return {"queries": queries}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

