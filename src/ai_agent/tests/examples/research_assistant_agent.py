"""Exemplo básico de agente assistente de pesquisa (esqueleto)."""

# Exemplo de uso ilustrativo — não espera-se que rode sem dependências adicionais.
from ai_agent.plugins.examples.web_scraper import WebScraperPlugin
from ai_agent.plugins.examples.file_manager import FileManagerPlugin

def run_demo():
    fm = FileManagerPlugin(root='.')
    ws = WebScraperPlugin()
    print('Arquivos atuais (exemplo):', fm.list_files('.'))
    print('Exemplo de fetch (não executa aqui):', 'ws.fetch("https://example.com")')

if __name__ == '__main__':
    run_demo()
