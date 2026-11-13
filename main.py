from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import Field,BaseModel
from base_agent import My_Agent,AIResponse
from common import llm

class RequestBody(BaseModel):

    question: str = Field(description="问题")


app = FastAPI()

# Allow the plain static frontend to call the API during local development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5500", "http://127.0.0.1:5500"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

llm = llm

my_agent = My_Agent(llm=llm)

@app.get('/')
async def hello():
    return 'hello world'


@app.get('/v1/{query}')
async def do_short_query(query: str):
    return query


@app.post('/v1')
async def query(request_body:RequestBody):

    if request_body is None or request_body.question is None:

        return AIResponse(answer="请先输入问题！")

    
    return my_agent.invoke(request_body.question)
