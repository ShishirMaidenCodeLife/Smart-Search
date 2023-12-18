from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
import ML_training.inference as ML_inference
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins=[
    "http://localhost",
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("auto_fill3.html", "r") as myHTML:
        index_html=myHTML.read()
        return HTMLResponse(content=index_html, status_code=200)

search_history_db=[]

## Uncomment below to just append the new search in previous history....
# @app.post("/process_form/")
# async def search(new_search: str = Form(...)):
#     search_history_db.append(new_search)
#     search_history_db.count
#     return{"thanks"}

## this is for appending + providing the https link prediction....
@app.post("/process_form/")
async def search(new_search: str = Form(...)):
    search_history_db.append(new_search)
    # search_history_db.count
    output=ML_inference.SVC_inference(new_search)
    output_link=output[0]
    return{"output":output_link}

@app.post("/search/")
async def search(new_search: str = Form(...)):

    item_counts = {}
    for item in search_history_db:
        if item in item_counts:
            item_counts[item] += 1
        else:
            item_counts[item] = 1

    # Print the counts
    for key, value in item_counts.items(): #changing the dict to list object using dict.items() and taking the key and value....
        print(f"Count of {key}: {value}")

    remove_redunt=set(search_history_db)
    new_search_history_db=list(remove_redunt)
    suggestions = [suggest for suggest in new_search_history_db if suggest.lower().startswith(new_search.lower())]
    # search_history_db.append(new_search)
    return {"suggestions": suggestions for i in range (1,6)}

@app.post("/test/")
async def test(new_search: str = Form(...)):
    return{"hello":"world"}