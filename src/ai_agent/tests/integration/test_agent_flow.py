"""Integração mínima: verifica fluxo entre componentes simples."""

import asyncio
from ai_agent.learning.experience_buffer import ExperienceBuffer

def test_experience_buffer_add_and_sample():
    buf = ExperienceBuffer(capacity=10)
    for i in range(5):
        buf.add({'id': i, 'skill': 'test'})
    assert buf.size() == 5
    s = buf.sample(2)
    assert len(s) == 2
