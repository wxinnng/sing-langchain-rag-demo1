from fastapi import FastAPI


app = FastAPI()


@app.get('/')
async def hello():

    return 'hello world'

@app.get('/v1/{query}')
async def do_short_query(query: str):

    return query
