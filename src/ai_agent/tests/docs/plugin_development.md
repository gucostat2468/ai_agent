# Desenvolvimento de Plugins

Regras básicas:
- Herdar de `ai_agent.plugins.base_plugin.BasePlugin`.
- Implementar `activate(context)`, `deactivate()` e `execute(...)`.
- Plugins de exemplo ficam em `ai_agent/plugins/examples/`.
- Evitar operações que escrevam fora da pasta definida pelo plugin (usar root path).
