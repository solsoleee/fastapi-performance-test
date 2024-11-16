from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pickle
import pandas as pd

# FastAPI 애플리케이션 초기화
app = FastAPI()

# 템플릿 설정
templates = Jinja2Templates(directory="templates")

# 모델과 벡터라이저 로드
with open("model_and_vectorizer.dump", "rb") as f:
    loaded_data = pickle.load(f)

loaded_model = loaded_data['model']
loaded_vectorizer = loaded_data['vectorizer']

# 메인 페이지 (HTML 입력창)
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# POST 엔드포인트: 예측
@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    title: str = Form(...),
    content: str = Form(...)
):
    if not title or not content:
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "error": "Invalid input. Provide both 'title' and 'content'."}
        )

    # 제목과 내용을 결합
    combined_text = title + " " + content
    new_data = pd.DataFrame({"data": [combined_text]})

    # 벡터화 및 예측
    X_new_tfidf = loaded_vectorizer.transform(new_data['data'])
    res = loaded_model.predict(X_new_tfidf)
    percentage = loaded_model.predict_proba(X_new_tfidf)[0]
    percentage0 = percentage[0]
    percentage1 = percentage[1]

    # 결과 HTML 렌더링
    return templates.TemplateResponse(
        "result.html",
        {
            "request": request,
            "prediction": res[0],
            "percentage0": percentage0,
            "percentage1": percentage1,
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
