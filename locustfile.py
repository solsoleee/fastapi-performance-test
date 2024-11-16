from locust import HttpUser, task, between
import random

class FastAPITestUser(HttpUser):
    wait_time = between(1, 2)  # 각 요청 사이의 대기 시간 (초)

    # 테스트 데이터 준비
    titles = ["Breaking News", "Sports Update", "Tech Innovations"]
    contents = [
        "The latest news about the economy.",
        "Highlights from yesterday's match.",
        "AI and ML are transforming industries."
    ]

    @task
    def test_predict(self):
        # 랜덤 데이터 생성
        title = random.choice(self.titles)
        content = random.choice(self.contents)

        # /predict 엔드포인트 요청
        response = self.client.post(
            "/predict",
            data={"title": title, "content": content}  # FastAPI는 Form 데이터를 지원
        )
        
        # 로깅
        if response.status_code == 200:
            print(f"Request successful: {response.json()}")
        else:
            print(f"Request failed with status code: {response.status_code}")
