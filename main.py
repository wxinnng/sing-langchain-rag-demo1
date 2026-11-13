from fastapi import FastAPI
from pydantic import Field,BaseModel
from base_agent import My_Agent,AIResponse
from common import llm

class RequestBody(BaseModel):

    question: str = Field(description="问题")


app = FastAPI()

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
