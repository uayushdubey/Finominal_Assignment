import sys
import os
import pytest
from fastapi.testclient import TestClient

# Ensure the root of the project is in python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from main import app
from repositories.data_repository import DataRepository

@pytest.fixture(scope="session")
def api_client():
    """
    FastAPI TestClient fixture for endpoint integration testing.
    """
    return TestClient(app)

@pytest.fixture(scope="session")
def repository():
    """
    DataRepository fixture for testing data ingestion pipelines.
    """
    return DataRepository()
