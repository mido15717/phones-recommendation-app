from fastapi import FastAPI
app = FastAPI()

@app.get("/welcome")
def welcomeapp():
    return {"message": "Welcome to the Phones Recommendation"}