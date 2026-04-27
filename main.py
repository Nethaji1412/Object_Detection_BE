import os

import cv2
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from pydantic import BaseModel
from ultralytics import YOLO

load_dotenv()

API_KEY = os.getenv("API_KEY", "secret-key")
MODEL_PATH = os.getenv("MODEL_PATH", "best.pt")
DATA_PATH = os.getenv("DATA_PATH", "nutrition_dataset_large.csv")

app = FastAPI(title="Food Nutrition API")

model = YOLO(MODEL_PATH)
df = pd.read_csv(DATA_PATH)

food_map = {
    "pizza_slice": "pizza",
    "burger_big": "burger"
}


def get_nutrition(food_list):
    nutrition_list = []
    for food in food_list:
        food = food_map.get(food, food)
        match = df[df["food_name"].str.lower() == food.lower()]
        if not match.empty:
            nutrition_list.append(match.iloc[0].to_dict())
    return nutrition_list


def calculate_total(nutrition_list):
    total = {"calories": 0, "protein": 0, "fat": 0, "carbs": 0}
    for item in nutrition_list:
        total["calories"] += item.get("calories", 0)
        total["protein"] += item.get("protein", 0)
        total["fat"] += item.get("fat", 0)
        total["carbs"] += item.get("carbs", 0)
    return total


def generate_summary(total):
    summary = ""
    if total["calories"] > 700:
        summary += "High calorie meal. "
    else:
        summary += "Balanced calories. "
    if total["protein"] < 15:
        summary += "Low protein. "
    else:
        summary += "Good protein level. "
    if total["fat"] > 20:
        summary += "High fat content. "
    return summary


def verify_api_key(x_api_key: str | None = Header(default=None)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return x_api_key


class PredictResponse(BaseModel):
    detected_foods: list[str]
    nutrition_data: list[dict]
    total: dict
    summary: str


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
async def predict(
    image: UploadFile = File(...),
    x_api_key: str = Depends(verify_api_key),
):
    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty image file")

    image_np = np.frombuffer(image_bytes, np.uint8)
    image_data = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
    if image_data is None:
        raise HTTPException(status_code=400, detail="Unable to decode image")

    results = model.predict(image_data)
    detected_foods = []
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]
            detected_foods.append(label)

    detected_foods = list(dict.fromkeys(detected_foods))
    nutrition_data = get_nutrition(detected_foods)
    total = calculate_total(nutrition_data)
    summary = generate_summary(total)

    return {
        "detected_foods": detected_foods,
        "nutrition_data": nutrition_data,
        "total": total,
        "summary": summary,
    }
