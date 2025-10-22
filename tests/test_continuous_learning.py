from typing import Any, Dict, List

import pandas as pd
import pytest

from src.models import training


class DummyCursor:
    def __init__(self, dictionary: bool = False):
        self.dictionary = dictionary
        self.executed: List[tuple[str, tuple[Any, ...] | None]] = []
        self._fetchone = (0,)
        self._fetchall: List[Dict[str, Any]] = []
        self.lastrowid = 0

    def execute(self, sql: str, params: tuple[Any, ...] | None = None) -> None:
        self.executed.append((sql.strip(), params))
        if sql.strip().upper().startswith('SELECT'):
            return
        if sql.strip().upper().startswith('INSERT'):
            self.lastrowid += 1

    def fetchone(self):
        return self._fetchone

    def fetchall(self):
        return self._fetchall

    def close(self) -> None:
        return None


class DummyConnection:
    def __init__(self, include_db: bool):
        self.include_db = include_db
        self.cursors: List[DummyCursor] = []
        self.autocommit = False

    def cursor(self, dictionary: bool = False):
        cursor = DummyCursor(dictionary=dictionary)
        self.cursors.append(cursor)
        return cursor

    def commit(self) -> None:
        return None

    def close(self) -> None:
        return None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


@pytest.fixture()
def dummy_connections(monkeypatch):
    connections: List[DummyConnection] = []

    def factory(config=None, include_db: bool = True):
        conn = DummyConnection(include_db)
        connections.append(conn)
        return conn

    monkeypatch.setattr('src.data.database.get_connection', factory)
    return connections


def test_database_schema(dummy_connections):
    from src.data import database

    database.initialize_database()

    first_conn = dummy_connections[0]
    assert not first_conn.include_db
    assert any('CREATE DATABASE IF NOT EXISTS' in sql for sql, _ in first_conn.cursors[0].executed)

    second_conn = dummy_connections[1]
    executed_sql = '\n'.join(sql for cursor in second_conn.cursors for sql, _ in cursor.executed)
    assert 'CREATE TABLE IF NOT EXISTS submissions' in executed_sql
    assert 'CREATE TABLE IF NOT EXISTS model_versions' in executed_sql
    assert 'CREATE TABLE IF NOT EXISTS prediction_logs' in executed_sql


def test_submission_workflow(dummy_connections):
    from src.data import database

    database.insert_submission(
        tweet_text='Delayed flight again',
        predicted_sentiment='negative',
        prediction_confidence=0.82,
        assigned_airline='Delta',
        true_sentiment=None,
    )
    insert_cursor = dummy_connections[-1].cursors[-1]
    assert insert_cursor.lastrowid == 1

    database.update_submission_status(
        submission_id=5,
        status='approved',
        true_sentiment='negative',
        admin_comment='Confirmed complaint',
    )
    update_cursor = dummy_connections[-1].cursors[-1]
    update_sql, update_params = update_cursor.executed[-1]
    assert 'UPDATE submissions' in update_sql
    assert update_params[-1] == 5


def test_retraining(monkeypatch):
    sample_original = pd.DataFrame(
        {
            'text': [
                'great flight', 'loved service', 'awesome crew', 'pleasant trip', 'smooth boarding',
                'bad flight', 'terrible delay', 'lost luggage', 'late departure', 'rude staff',
                'average trip', 'decent flight', 'okay service', 'acceptable ride', 'mediocre snack'
            ],
            'airline_sentiment': [
                'positive', 'positive', 'positive', 'positive', 'positive',
                'negative', 'negative', 'negative', 'negative', 'negative',
                'neutral', 'neutral', 'neutral', 'neutral', 'neutral'
            ],
        }
    )

    monkeypatch.setattr('src.data.loaders.load_normalized_dataset', lambda: sample_original)

    new_samples = [
        {'tweet_text': 'fantastic crew', 'sentiment': 'positive'},
        {'tweet_text': 'terrible delay', 'sentiment': 'negative'},
        {'tweet_text': 'mediocre service', 'sentiment': 'neutral'},
    ]

    pipeline, results = training.retrain_with_new_data(new_samples, model_name='linear_svc')

    assert results['new_samples_added'] == 3
    assert 'classification_report' in results
    assert hasattr(pipeline, 'predict')


def test_feature_importance():
    pipeline = training.load_trained_model()
    explanation = training.explain_prediction(pipeline, 'Great flight!')

    assert explanation['sentiment'] in {'positive', 'neutral', 'negative'}
    assert 'reasoning' in explanation
    assert isinstance(explanation['top_features'], list)
    assert len(explanation['top_features']) > 0

    batch = training.batch_explain_predictions(pipeline, ['Great flight!', 'Terrible service'])
    assert len(batch) == 2
    assert all('sentiment' in item for item in batch)
