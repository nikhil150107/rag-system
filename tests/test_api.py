"""Integration tests for Flask API endpoints with offline mocked LLM."""
import pytest
from unittest.mock import MagicMock, patch
import io
import uuid
import app as flask_app_module


@pytest.fixture
def client():
    flask_app_module.app.config["TESTING"] = True
    with flask_app_module.app.test_client() as client:
        yield client


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    json_data = response.get_json()
    assert json_data["status"] == "ok"
    assert "components" in json_data
    assert json_data["components"]["vectorstore"] == "ok"
    assert json_data["components"]["embedding_model"] == "ok"
    assert json_data["components"]["reranker"] == "ok"


def test_metrics_endpoint(client):
    response = client.get("/metrics")
    assert response.status_code == 200
    json_data = response.get_json()
    assert "total_requests" in json_data
    assert "context_found_rate" in json_data
    assert "fallback_rate" in json_data
    assert "avg_total_latency_ms" in json_data


def test_upload_txt_file_and_duplicate(client):
    unique_id = uuid.uuid4().hex
    content = f"Unique document content {unique_id} for API upload verification.".encode("utf-8")
    filename = f"test_api_doc_{unique_id}.txt"

    # First upload -> should be new (duplicate: False)
    data = {
        "file": (io.BytesIO(content), filename)
    }
    res1 = client.post("/upload", data=data, content_type="multipart/form-data")
    assert res1.status_code == 200
    json1 = res1.get_json()
    assert json1["success"] is True
    assert json1["duplicate"] is False
    assert json1["filename"] == filename
    assert json1["chunks_created"] >= 1

    # Second upload of the exact same content -> duplicate protection
    data_dup = {
        "file": (io.BytesIO(content), filename)
    }
    res2 = client.post("/upload", data=data_dup, content_type="multipart/form-data")
    assert res2.status_code == 200
    json2 = res2.get_json()
    assert json2["success"] is True
    assert json2["duplicate"] is True
    assert "already indexed" in json2["message"]
    assert json2["chunks_created"] == 0


def test_ask_endpoint_empty_question(client):
    res = client.post("/ask", json={"question": ""})
    assert res.status_code == 400
    assert "required" in res.get_json()["error"]


@patch.object(flask_app_module, "get_openai_client")
def test_ask_endpoint_mocked_llm(mock_get_client, client):
    mock_openai = MagicMock()
    mock_completion = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = "Mocked LLM answer based on context."
    mock_completion.choices = [mock_choice]
    mock_openai.chat.completions.create.return_value = mock_completion
    mock_get_client.return_value = mock_openai

    res = client.post("/ask", json={"question": "What is in test_api_doc?"})
    assert res.status_code == 200
    json_data = res.get_json()
    assert "request_id" in json_data
    assert json_data["request_id"].startswith("req_")
    assert "context_found" in json_data
    assert "answer" in json_data
    assert "search_query" in json_data
    assert json_data["answer"] == "Mocked LLM answer based on context."


@patch.object(flask_app_module, "get_openai_client")
def test_ask_endpoint_conversational_multi_turn(mock_get_client, client):
    """Verify /ask with conversation_history reformulates query and returns answer."""
    mock_openai = MagicMock()

    # First completion call is for query reformulation, second is for final generation
    mock_choice_reform = MagicMock()
    mock_choice_reform.message.content = '{"search_query": "What is the price of the Pro plan?"}'
    mock_resp_reform = MagicMock(choices=[mock_choice_reform])

    mock_choice_ans = MagicMock()
    mock_choice_ans.message.content = "The Pro plan costs $20/month."
    mock_resp_ans = MagicMock(choices=[mock_choice_ans])

    mock_openai.chat.completions.create.side_effect = [mock_resp_reform, mock_resp_ans]
    mock_get_client.return_value = mock_openai

    payload = {
        "question": "How much does it cost?",
        "conversation_history": [
            {"role": "user", "content": "Tell me about the Pro plan."},
            {"role": "assistant", "content": "The Pro plan includes advanced analytics."}
        ]
    }

    res = client.post("/ask", json=payload)
    assert res.status_code == 200
    json_data = res.get_json()
    assert "request_id" in json_data
    assert json_data["request_id"].startswith("req_")
    assert json_data["search_query"] == "What is the price of the Pro plan?"
    assert json_data["answer"] == "The Pro plan costs $20/month."
    assert "context_found" in json_data
    assert "sources" in json_data

