"""Script de migração de dados fictício para ambiente de desenvolvimento."""

from pathlib import Path

def migrate_to_sqlite(db_path: str = './ai_agent.db'):
    p = Path(db_path)
    if not p.exists():
        p.write_text('-- sqlite db placeholder --')
    print('Migração (simulada) concluída:', db_path)

if __name__ == '__main__':
    migrate_to_sqlite()
