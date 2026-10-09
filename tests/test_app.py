import importlib
import json
from io import BytesIO

import pytest
from fastapi.testclient import TestClient
from PIL import Image

backend = importlib.import_module('prompting_pot.app')


@pytest.fixture
def client(tmp_path, monkeypatch):
    config = tmp_path / 'config'
    config.mkdir()
    (config / 'targets.json').write_text(json.dumps([
        {'id': 'vision', 'name': 'Vision', 'config': 'vision.yaml', 'vision': True},
        {'id': 'text', 'name': 'Text', 'config': 'text.yaml', 'vision': False},
    ]))
    monkeypatch.setattr(backend, 'CONFIG_DIR', config)

    async def ready(target):
        return 'http://localhost:8080/v1'

    monkeypatch.setattr(backend, 'ensure_llama_swap', ready)
    return TestClient(backend.app)


def test_page_and_targets(client):
    assert client.get('/').status_code == 200
    assert client.get('/static/app.js').status_code == 200
    targets = client.get('/api/targets').json()
    assert len(targets) == 2
    assert 'config_path' not in targets[0]


@pytest.mark.parametrize('data', [
    {'target_id': 'missing', 'mode': 'text', 'description': 'hello'},
    {'target_id': 'text', 'mode': 'invalid'},
    {'target_id': 'text', 'mode': 'text', 'description': '  '},
    {'target_id': 'vision', 'mode': 'image'},
    {'target_id': 'text', 'mode': 'image'},
])
def test_invalid_input(client, data):
    assert client.post('/api/generate', data=data).status_code == 422


def test_unconnected_backend(client):
    response = client.post('/api/generate', data={'target_id': 'text', 'mode': 'text', 'description': 'tea shop'})
    assert response.status_code == 501
    assert 'not connected' in response.json()['detail']


@pytest.mark.parametrize('output', ['a cozy tea shop', {'prompt': 'tea shop'}])
def test_wired_text_contract(client, monkeypatch, output):
    async def start(target):
        return 'http://localhost:8080/v1'

    async def infer(**kwargs):
        assert kwargs['description'] == 'tea shop'
        assert kwargs['image'] is None
        return output

    monkeypatch.setattr(backend, 'ensure_llama_swap', start)
    monkeypatch.setattr(backend, 'generate_prompt', infer)
    response = client.post('/api/generate', data={'target_id': 'text', 'mode': 'text', 'description': ' tea shop '})
    assert response.json() == {'output': output}


def test_image_contract(client, monkeypatch):
    image = BytesIO()
    Image.new('RGB', (2, 2)).save(image, format='PNG')

    async def start(target):
        return 'http://localhost:8080/v1'

    async def infer(**kwargs):
        assert kwargs['image'] == image.getvalue()
        assert kwargs['image_content_type'] == 'image/png'
        assert kwargs['description'] == 'make it pink'
        assert kwargs['image_generator'] == 'krea2'
        assert kwargs['prompt'] == backend.build_image_prompt('krea2')
        return {'prompt': 'pink'}

    monkeypatch.setattr(backend, 'ensure_llama_swap', start)
    monkeypatch.setattr(backend, 'generate_prompt', infer)
    response = client.post('/api/generate', data={'target_id': 'vision', 'mode': 'image', 'description': 'make it pink', 'image_generator': 'krea2'}, files={'image': ('image.png', image.getvalue(), 'image/png')})
    assert response.status_code == 200


def test_invalid_image(client):
    response = client.post('/api/generate', data={'target_id': 'vision', 'mode': 'image', 'image_generator': 'krea2'}, files={'image': ('fake.png', b'not an image', 'image/png')})
    assert response.status_code == 422


def test_oversize_image(client):
    response = client.post('/api/generate', data={'target_id': 'vision', 'mode': 'image', 'image_generator': 'krea2'}, files={'image': ('big.png', b'x' * (backend.MAX_IMAGE_BYTES + 1), 'image/png')})
    assert response.status_code == 422


def test_config_path_escape(client):
    (backend.CONFIG_DIR / 'targets.json').write_text(json.dumps([
        {'id': 'bad', 'name': 'Bad', 'config': '../bad.yaml', 'vision': True},
    ]))
    with pytest.raises(ValueError):
        backend.load_targets()
