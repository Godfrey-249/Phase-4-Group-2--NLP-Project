from fastapi import FastAPI
from pydantic import BaseModel

class Item(BaseModel):
    title: str
    description: str

app = FastAPI()

@app.post("/items/")
async def create_item(item: Item):
    # Handle the data (e.g. save to database)
    return {"message": "Received!", "item": item}
