"""Exemplo de agente para automação de tarefas (esqueleto)."""

from ai_agent.plugins.examples.file_manager import FileManagerPlugin

def run_task():
    fm = FileManagerPlugin(root='.')
    files = fm.list_files('.')
    print(f"Encontrados {len(files)} ficheiros na raiz do projeto (exemplo)")
