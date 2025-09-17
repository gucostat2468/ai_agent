"""Exemplo mínimo: iniciar um agente usando um LLM 'mock'."""

import asyncio
import sys
import logging

# adiciona src ao path se necessário (quando rodar como script)
from pathlib import Path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root / 'src'))

from ai_agent.core.agent import AIAgent  # may be partial; this example uses a mock
from ai_agent.interfaces.base import LLMInterface
from ai_agent.config.models import BaseConfig


class DummyLLM(LLMInterface):
    async def generate(self, prompt: str, **kwargs):
        # retorno simples de eco para demonstração
        return f"ECHO: {prompt}"

async def main():
    config = BaseConfig()
    llm = DummyLLM()
    agent = AIAgent(config=config, llm_interface=llm)
    # many implementations will provide an async start, guard with getattr
    if hasattr(agent, 'start'):
        try:
            await agent.start()
        except Exception:
            logging.info('Agent start skipped in minimal example')
    resp = await agent.process_message("Olá agente!", None)
    print("Resposta do agente:", resp)

if __name__ == '__main__':
    asyncio.run(main())
